# %%writefile train_ddp.py
   
import io
import os
from time import time
from time import sleep
from typing import Union, Callable, Optional

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import torchvision as tv
import pandas as pd
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

from dataclasses import dataclass
import wandb
import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# # preferences
# def enable_determinism():
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#     torch.use_deterministic_algorithms(True) # на этот раз зафиксируем алгоритмы, чтобы изменения точно не были случайными

def fix_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.mps.manual_seed(seed)
    
def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # любой свободный порт
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# dataset
class TinyImageNetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Callable):
        super().__init__()

        self._data = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._data.iloc[idx]

        image = Image.open(io.BytesIO(sample['image.bytes']))
        if image.mode == 'L':
            image = image.convert('RGB')
        image = self.transform(image)

        label = torch.tensor(sample['label'], dtype=torch.long)
        return image, label

# train-val split
def stratified_train_val_split(df: pd.DataFrame, train_share: float, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)

    label_counts = df['label'].value_counts() # посчитаем число лейблов каждого класса
    train_counts = (label_counts * train_share).round().astype(int) # посчитаем, какая часть лейблов в каждом классе пойдёт на train

    train_indices, val_indices = [], []
    for label in label_counts.index:
        class_indices = df[df['label'] == label].index # получим индексы всех семплов данного класса
        
        shuffled_indices = np.random.permutation(class_indices) # перемешаем
        
        n_train = train_counts[label] # сколько должно быть семплов этого класса в трейне
        train_indices.extend(shuffled_indices[:n_train]) # первые n_train идут в train
        val_indices.extend(shuffled_indices[n_train:]) # остаток — в val
    
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    
    train_df.sort_index(inplace=True)
    val_df.sort_index(inplace=True)

    return train_df, val_df

# net blocks
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, squeeze_rate: int) -> None:
        super().__init__()
        
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // squeeze_rate, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // squeeze_rate, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se_block(x)
        return scale * x
        

class MBConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, exp_channels: int, kernel_size: Union[int, tuple[int, int]],
                 padding: Union[int, tuple[int, int]], stride: Union[int, tuple[int, int]], non_linearity: str = 'RE',
                 se_block: bool = True, squeeze_rate: int = 16) -> None:
        super().__init__()

        self.non_linearity_bank = {'RE': nn.ReLU6, 'HS': nn.Hardswish}

        self.use_skip_connection = stride != 2 and in_channels == out_channels

        self.layers = []

        if exp_channels != in_channels:
            self.layers.append(nn.Conv2d(in_channels, exp_channels, kernel_size=1))
            self.layers.append(nn.BatchNorm2d(exp_channels))
            self.layers.append(self.non_linearity_bank[non_linearity]())
        
        self.layers.append(nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, padding, groups=exp_channels))
        self.layers.append(nn.BatchNorm2d(exp_channels))
        self.layers.append(self.non_linearity_bank[non_linearity]())
        
        if se_block:
            self.layers.append(SqueezeExcitation(exp_channels, squeeze_rate))
        
        self.layers.append(nn.Conv2d(exp_channels, out_channels, kernel_size=1))
        self.layers.append(nn.BatchNorm2d(out_channels))
        
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        # Если пространственный размер входного тензора не меняется, то прибавляем скип
        if self.use_skip_connection:
            out = out + x
        return out

# main model
class MobileNetV3(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        self.feature_extractor = nn.Sequential(
            # bneck  in  out  exp k  p  s     NE    se
            MBConv2d(16, 16, 16, 3, 1, 1, 'RE', True),
            MBConv2d(16, 24, 64, 3, 1, 2, 'RE', False),
            MBConv2d(24, 24, 72, 3, 1, 1, 'HS', True),
            MBConv2d(24, 32, 72, 3, 1, 1, 'HS', True),
            MBConv2d(32, 64, 96, 3, 1, 2, 'HS', True),
            MBConv2d(64, 64, 128, 3, 1, 1, 'HS', True),
            MBConv2d(64, 128, 128, 3, 1, 1, 'HS', True),
            MBConv2d(128, 128, 256, 3, 1, 1, 'HS', True),
            MBConv2d(128, 128, 256, 3, 1, 1, 'HS', True),
            nn.Conv2d(128, 512, 1),
            nn.BatchNorm2d(512),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Hardswish(),
            nn.Linear(1024, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.classifier(x)
        return x

# mixing augmentations
def MixUp(batch, alpha=0.2):
    """
    Берем по 2 случайные картинки внутри батча и смешиваем их в некоторой пропорции
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    images_shuffled = images[index]
    labels_shuffled = labels[index]

    mixed_images = lam * images + (1 - lam) * images_shuffled
    return mixed_images, labels, labels_shuffled, lam


def MixCut(batch, alpha=0.2):
    """
    Делаем вставку одной картинки в другой
    """
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = images.size(0)
    index = torch.randperm(batch_size)
    
    images_shuffled = images[index]
    labels_shuffled = labels[index]
    
    batch_size, channels, height, width = images.shape

    # Стандартная формула для CutMix
    cut_ratio = np.sqrt(1 - lam)  
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)
    
    # Генерируем координаты
    cx = np.random.randint(0, width)
    cy = np.random.randint(0, height)
    
    # Вычисляем границы
    x1 = np.clip(cx - cut_w // 2, 0, width)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    
    # Создаем маску
    mask = torch.ones((height, width), dtype=torch.float32)
    mask[y1:y2, x1:x2] = 0
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Смешиваем изображения
    mixed_images = images * mask + images_shuffled * (1 - mask)
    
    # Пересчитываем lambda
    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1)) / (width * height)
    
    return mixed_images, labels, labels_shuffled, lam_adjusted

def collate_fn(batch, alpha=0.2, choice_mixup=True):
    """
    True - MixUp
    False - MixCut
    None - 50/50
    """
    if choice_mixup is None:
        choice_mixup = np.random.random() > 0.5

    # 50/50
    if choice_mixup:
        return MixUp(batch, alpha)
    else:
        return MixCut(batch, alpha)


def mixup_cutmix_criterion(criterion, pred, labels_a, labels_b, lam):
    """
    Вычисление loss для MixUp/CutMix
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


# training usinng Distributed Data Parallel
def run_epoch_ddp(model, epoch, loader, criterion, optimizer=None, scheduler=None, 
                  device=torch.device("cpu"), rank=0, scaler=None):
    all_labels, all_preds = [], []
    loss_epoch = 0.
    
    for batch in tqdm.tqdm(loader, disable=rank != 0):
        # Обработка mixup/cutmix или обычных данных
        if isinstance(batch, tuple) and len(batch) == 4:
            images, labels_a, labels_b, lam = batch
            images, labels_a, labels_b = images.to(device), labels_a.to(device), labels_b.to(device)

            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = mixup_cutmix_criterion(criterion, logits, labels_a, labels_b, lam)
                
            preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            all_labels.extend(labels_a.cpu().numpy())
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)
                
            preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            all_labels.extend(labels.cpu().numpy())

        if optimizer is not None:
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()

        loss_epoch += loss.item()
        all_preds.extend(preds.cpu().numpy())

    loss_epoch /= len(loader)
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    acc_epoch = (all_preds == all_labels).sum() / len(all_preds)

    if rank == 0:
        if optimizer is not None:
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)    
            wandb.log({'train loss': loss_epoch}, step=epoch)
            wandb.log({'train accuracy': acc_epoch * 100.0}, step=epoch)
        else:
            wandb.log({'test loss': loss_epoch}, step=epoch)
            wandb.log({'test accuracy': acc_epoch * 100.0}, step=epoch)
    return loss_epoch, acc_epoch

def train_ddp(model, n_epochs, train_loader, criterion, optimizer, scheduler=None, 
              val_loader=None, val_freq=10, save_best=True, save_name='model', 
              device=torch.device("cpu"), rank=0, scaler=None):
    enable_validation = val_loader is not None
    best_val = 0.
    
    for epoch in range(n_epochs):
        timer_start = time()
        model.train()
        train_sampler = train_loader.sampler
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)  # 🔑 Важно для DDP: перемешивание данных между эпохами
            
        train_loss, train_acc = run_epoch_ddp(model, epoch, train_loader, criterion, optimizer, scheduler, device, rank, scaler=scaler)

        if rank == 0:
            print(f"Epoch {epoch+1}: Train loss: {train_loss:.4f} | Train acc: {train_acc*100:.2f}%")

        if enable_validation and epoch % val_freq == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_acc = run_epoch_ddp(model, epoch, val_loader, criterion, 
                                                  optimizer=None, scheduler=None, device=device, rank=rank, scaler=scaler)
            if rank == 0:
                print(f"Val loss: {val_loss:.4f} | Val acc: {val_acc*100:.2f}%")
                
            if rank == 0 and save_best and val_acc >= best_val:
                best_val = val_acc
                # Сохраняем state_dict без DDP обёртки
                torch.save(model.module.state_dict(), f"{save_name}.pth")
                print(f"💾 Saved best model at epoch {epoch+1}")

        if rank == 0:
            print(f"⏱ Time spent on epoch: {time() - timer_start:.2f}s")
            
    return model

# config
@dataclass
class Config:
    seed: int = 24
    batch_size: int = 1000
    img_size: int = 64
    n_epochs: int = 20
    lr: float = 3e-4

def main():
    # 🔑 torchrun автоматически выставляет эти переменные
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, device_id=local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    config = Config()
    # ✅ Заменяем enable_determinism() на безопасный вариант для DDP
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    torch.cuda.manual_seed(config.seed + rank)
    
    # --- Загрузка данных (в каждом процессе) ---
    data_path = "/kaggle/input/datasets/andreykurdyubov/tiny-imagenet/tiny_imagenet/train-00000-of-00001-1359597a978bc4fa.parquet"
    cache_path = "/tmp/train_cached.parquet"
    if rank == 0 and not os.path.exists(cache_path):
        try:
            print(f"💾 Caching dataset to {cache_path}...")
            df_orig = pd.read_parquet(data_path, engine='fastparquet')
            if 'image.path' in df_orig.columns:
                df_orig.drop(columns=['image.path'], inplace=True)
            df_orig.to_parquet(cache_path, engine='fastparquet')
        except Exception as e:
            print(f"⚠️ Cache failed: {e}")
            
    # Ждём появления файла (безопаснее, чем dist.barrier() при файловых операциях)
    timeout = 30
    start_wait = time()
    while not os.path.exists(cache_path):
        if time() - start_wait > timeout:
            raise RuntimeError(f"Cache file not created within {timeout}s")
        sleep(0.1)
    
    df = pd.read_parquet(cache_path, engine='fastparquet')
    # df = pd.read_parquet(data_path, engine='fastparquet')
    # df.drop(columns=['image.path'], inplace=True)
    train_df, val_df = stratified_train_val_split(df, train_share=0.9, seed=config.seed)
    
    # transforms
    train_transform = transforms.Compose([
        # transforms.RandAugment(num_ops=2, magnitude=9),
        # transforms.TrivialAugmentWide(),
        # Transforms автора
        transforms.RandomCrop(size=(56, 56)),
        transforms.RandomHorizontalFlip(0.5),
        # transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.PILToTensor(),
        transforms.ToDtype(dtype=torch.float32, scale=True),
        transforms.RandomErasing()
    ])

    val_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ToDtype(dtype=torch.float32, scale=True)
    ])
    
    train_ds = TinyImageNetDataset(train_df, train_transform)
    val_ds = TinyImageNetDataset(val_df, val_transform)
    
    # 🔑 Samplers вместо shuffle=True
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True, 
        seed=config.seed,
     )
    
    val_sampler = DistributedSampler(
        val_ds, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False, 
        seed=config.seed
    )
    
    # 🔑 shuffle=False, т.к. порядок задаёт Sampler
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        sampler=train_sampler,
        num_workers=4, 
        pin_memory=False,
        persistent_workers=True,
        drop_last=True, 
        worker_init_fn=seed_worker,
        # collate_fn=lambda batch: collate_fn(batch, alpha=0.3, choice_mixup=True),
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.batch_size, 
        sampler=val_sampler,                           
        num_workers=4, 
        pin_memory=True
    )
    
    model = MobileNetV3(3, 200).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.n_epochs * len(train_loader),
    eta_min=1e-5)
    
    if rank == 0:
        wandb.init(project="CV-hw3-TinyImageNet", name="DDP Simple Augs 4workers batch=1000", config=config.__dict__)
        
    model = train_ddp(model, config.n_epochs, train_loader, criterion, optimizer, scheduler,
              val_loader=val_loader, val_freq=1, save_best=True, save_name="model_ddp", 
              device=device, rank=rank, scaler=scaler)
    
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()

# в Kaggle c 2 T4
# !pip install fastparquet
# import wandb
# wandb.login()
# !torchrun --nproc_per_node=2 train_ddp.py