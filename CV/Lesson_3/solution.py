# !pip install fastparquet

import io
import os
from time import time
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

# фиксируем сиды
def enable_determinism():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True) # на этот раз зафиксируем алгоритмы, чтобы изменения точно не были случайными

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

def myshow(img):
    # img = img * 0.3 + 0.3 # умножаем на std, прибавляем mean - в Normalize всё наоборот
    npimg = img.detach().numpy()
    fig = plt.figure(figsize=(16, 16))
    plt.imshow(npimg.transpose(1, 2, 0))

def run_epoch(model: nn.Module, epoch, loader: DataLoader, criterion: Callable, optimizer: Optional[torch.optim.Optimizer] = None,\
              scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    all_labels, all_preds = [], []
    loss_epoch = 0.
    for batch in tqdm.tqdm(loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        loss_epoch += loss.item()
        preds = torch.argmax(logits.softmax(dim=-1), dim=-1)
        if len(labels.size()) > 1:
            labels = labels.argmax(dim=-1)

        all_preds = np.concatenate((all_preds, preds.cpu().numpy()))
        all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

    loss_epoch /= len(loader)
    acc_epoch = (all_preds == all_labels).sum() / len(all_preds)

    if optimizer is not None:
        wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)    
        wandb.log({'train loss': loss_epoch}, step=epoch)
        wandb.log({'train accuracy': acc_epoch * 100.0}, step=epoch)
    else:
        wandb.log({'test loss': loss_epoch}, step=epoch)
        wandb.log({'test accuracy': acc_epoch * 100.0}, step=epoch)

    return loss_epoch, acc_epoch

def train(model: nn.Module, n_epochs: int, train_loader: DataLoader, criterion: Callable, optimizer: torch.optim.Optimizer,
          scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None, val_loader: Optional[DataLoader] = None, val_freq: int = 10,\
          save_best: bool = True, save_name: str = 'model', device: torch.device = torch.device("cpu")) -> nn.Module:
    enable_validation = val_loader is not None
    best_val = 0.

    for epoch in range(n_epochs):
        timer_start = time()
        model.train()
        train_loss_epoch, train_acc_epoch = run_epoch(model, epoch, train_loader, criterion, optimizer, scheduler, device)

        print(f"Epoch {epoch+1}:")
        print(f"Train loss: {train_loss_epoch} | Train acc: {train_acc_epoch * 100}%")

        if enable_validation and epoch % val_freq == 0:
            model.eval()
            with torch.no_grad():
                val_loss_epoch, val_acc_epoch = run_epoch(model, epoch, val_loader, criterion, optimizer=None, scheduler=None, device=device)

            if save_best and val_acc_epoch >= best_val:
                best_val = val_acc_epoch
                model.to("cpu")
                torch.save(model.state_dict(), f"{save_name}.pth")
                model.to(device)

            print(f"Val loss: {val_loss_epoch} | Val acc: {val_acc_epoch * 100}%")

        print(f"Time spent on epoch: {time() - timer_start}")

    return model

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

model = MobileNetV3(3, 200)
model_summary = summary(model, input_size=(1,3,64,64), verbose=True)

assert model_summary.total_params <= 1.5e6, "Слишком много параметров, уменьшите сеть"
assert model_summary.total_mult_adds <= 1e8, "Слишком высокая вычислительная сложность, оптимизируйте сеть"

data_path = r"/kaggle/input/datasets/andreykurdyubov/tiny-imagenet/tiny_imagenet/train-00000-of-00001-1359597a978bc4fa.parquet" # замените на путь до .parquet файла с train частью датасета
df = pd.read_parquet(data_path, engine='fastparquet')
df.drop(columns=['image.path'], inplace=True)

train_df, val_df = stratified_train_val_split(df, train_share=0.9, seed=42)

# определите здесь CutMix/MixUp

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


def collate_fn(batch):
    choice = np.random.random() > 0.5
    if choice:
        return MixUp(batch, alpha=0.2)
    else:
        return MixCut(batch, alpha=0.2)
    
def mixup_cutmix_criterion(criterion, pred, labels_a, labels_b, lam):
    """
    Вычисление loss для MixUp/CutMix
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


def compute_metrics_for_cutmix(logits, labels_a, labels_b, lam):
    """
    Вычисление метрик для CutMix (более точный способ)
    """
    # Получаем вероятности
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    
    correct_a = (preds == labels_a).float()
    simple_accuracy = correct_a.mean().item()
    
    return simple_accuracy, preds


def run_epoch_cutmix(model: nn.Module, epoch, loader: DataLoader, criterion: Callable, optimizer: Optional[torch.optim.Optimizer] = None,\
              scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    all_labels, all_preds = [], []
    loss_epoch = 0.
    acc_epoch = 0.
    
    for batch in tqdm.tqdm(loader):
        if optimizer is not None:
            images, labels_a, labels_b, lam = batch
            images, labels_a, labels_b = images.to(device), labels_a.to(device), labels_b.to(device)
    
            logits = model(images)
            loss = mixup_cutmix_criterion(criterion, logits, labels_a, labels_b, lam)
            preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if scheduler is not None:
                scheduler.step()
    
            loss_epoch += loss.item()
            # simple_accuracy, preds = compute_metrics_for_cutmix(logits, labels_a, labels_b, lam)
    
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_a.cpu().numpy())
            
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
    
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
  
            loss_epoch += loss.item()
      
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    loss_epoch /= len(loader)
    acc_epoch = (all_preds == all_labels).sum() / len(all_preds)

    if optimizer is not None:
        wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)    
        wandb.log({'train loss': loss_epoch}, step=epoch)
        wandb.log({'train accuracy': acc_epoch * 100.0}, step=epoch)
    else:
        wandb.log({'test loss': loss_epoch}, step=epoch)
        wandb.log({'test accuracy': acc_epoch * 100.0}, step=epoch)

    return loss_epoch, acc_epoch

def train_cutmix(model: nn.Module, n_epochs: int, train_loader: DataLoader, criterion: Callable, optimizer: torch.optim.Optimizer,
          scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None, val_loader: Optional[DataLoader] = None, val_freq: int = 10,\
          save_best: bool = True, save_name: str = 'model', device: torch.device = torch.device("cpu")) -> nn.Module:
    enable_validation = val_loader is not None
    best_val = 0.

    for epoch in range(n_epochs):
        timer_start = time()
        model.train()
        train_loss_epoch, train_acc_epoch = run_epoch_cutmix(model, epoch, train_loader, criterion, optimizer, scheduler, device)

        print(f"Epoch {epoch+1}:")
        print(f"Train loss: {train_loss_epoch} | Train acc: {train_acc_epoch * 100}%")

        if enable_validation and epoch % val_freq == 0:
            model.eval()
            with torch.no_grad():
                val_loss_epoch, val_acc_epoch = run_epoch_cutmix(model, epoch, val_loader, criterion, optimizer=None, scheduler=None, device=device)

            if save_best and val_acc_epoch >= best_val:
                best_val = val_acc_epoch
                model.to("cpu")
                torch.save(model.state_dict(), f"{save_name}.pth")
                model.to(device)

            print(f"Val loss: {val_loss_epoch} | Val acc: {val_acc_epoch * 100}%")

        print(f"Time spent on epoch: {time() - timer_start}")

    return model

# конфиг с основными гиперпараметрами
@dataclass
class Config:
    seed: int = 24
    batch_size: int = 100
    img_size: int = 64
    n_epochs: int = 10
    lr: float = 1e-4

# не забывайте, что фиксировать заново сиды и создавать даталоадеры с ними нужно каждый раз, если хотите воспроизводимости
config = Config()
enable_determinism()
fix_seeds(config.seed)

generator = torch.Generator()
generator.manual_seed(config.seed)

# transforms
train_transform = transforms.Compose([
    # transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.PILToTensor(),
    transforms.ToDtype(dtype=torch.float32, scale=True)
])

val_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ToDtype(dtype=torch.float32, scale=True)
])

train_dataset = TinyImageNetDataset(train_df, train_transform)
val_dataset = TinyImageNetDataset(val_df, val_transform)

# Если используете MixUp/CutMix, не забудьте добавить в train_loader collate_fn=collate_fn
train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    num_workers=4, 
    shuffle=True, 
    pin_memory=False,
    drop_last=True,
    worker_init_fn=seed_worker,
    # collate_fn=lambda batch: collate_fn(batch, alpha=1, choice_mixup=False),
    generator=generator,
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=config.batch_size, 
    pin_memory=False, 
    shuffle=False)
                           

n_epochs = config.n_epochs
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MobileNetV3(3, 200)
# if torch.cuda.device_count() > 1:
#     print(f"Используется {torch.cuda.device_count()} видеокарты!")
#     # Оборачиваем модель в DataParallel
#     torch.use_deterministic_algorithms(False)
#     model = nn.DataParallel(model)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

wandb.init(
    project="CV-hw3-TinyImageNet", 
    name="No norms, no augs", 
    config=config.__dict__
)

model = train(model, n_epochs, train_loader, criterion, optimizer, scheduler=None, val_loader=val_loader, val_freq=1, save_best=False, save_name="model_p1", device=device)

wandb.finish()