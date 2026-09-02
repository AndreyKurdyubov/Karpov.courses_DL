### Курс [DL engineer: CV](https://karpov.courses/deep-learning) от Karpov Cources
### Мой [блог в ТГ](https://t.me/dl_journey)
### Старт прохождения 02.02.2026. Актуальный прогресс на 02.09.2026:
#### Часть 1. Base DL
<details><summary> 1. Обзор Deep Learning </summary> 
<pre>
* Теория
    - Устройство нейросетей. Нейрон. Объединение нейронов 
    - Обучение нейросетей. Градиентный спуск
    - Обратное распространение ошибки
    - Ускорение вычислений. Батчи. Использование видеокарт
* Практика
    - Основы работы в pyTorch 
    - Операции с тензорами
    - Градиенты
    - Функции потерь
    - Слои
    - Вычисление на GPU
</pre>
</details>

<details><summary> 2. Построение нейросети и методы оптимизации </summary> 
<pre>
* Теория
    - Роль нелинейности в нейросетях
    - Функция активации. Sigmoid, Tanh, ReLU
    - Архитектура нейросетей. Полносвязный слой
    - Backpropagation. Математические принципы
    - Градиентный спуск. SGD. AdaGrad. RMSProp. ADAM.
* Практика
    - Классификация изображений. NotMNIST A - J
    - Задача регрессии и оптимизатор Adam
    - Ручная реализация многослойной сети и градиентного спуска в numpy
</pre>
</details>

<details><summary> 3. Продвинутые техники обучения моделей </summary> 
<pre>
* Теория
    - Слой номализации BatchNorm1d
    - Слой Dropout
    - Learning Rate scheduler
    - Провение экспериментов
* Практика
    - Классификация предметов одежды из датасета FashionMNIST. Применение нормализации, dropout, scheduler
    - Выгрузка и визуализация метрик на wandb.ai
    - Сохранение и загрузка параметров модели
</pre>
</details>

<details><summary> 4. Основы компьютерного зрения (CV) </summary> 
<pre>
* Теория
    - Обзор типов задач CV
    - Операция свертки. Сверточный слой. Receptive field
    - Параметры свертки. kernel size, padding, stride, dilation
    - Методы уменьшения размерности. Pooling, flattering
    - Архитектуры сверточных сетей. Lenet VGG. Inception. ResNet (<a href="https://arxiv.org/abs/1512.03385">Deep Residual Learning</a>)
* Практика
    - Свертка, pooling
    - Разбиение на батчи. Dataset и Dataloader
    - Аугментация. RandomResizedCrop. Rotate. Hotizontal flip. Resize. Normalize
    - Построение сверточной сети для классификации объектов из датасета CIFAR10
    - Finetuning готовой модели на примере ResNet, mobilnet
    - Исследование косинусной схожести эмбедингов объектов одного класса
</pre>
</details>

<details><summary> 5. Типы задач CV </summary> 
<pre>
* Теория
    - Семантическая сегментация. Архитектура <a href="https://arxiv.org/abs/1505.04597v1">U-Net </a>
    - Детекция объектов
        1. Архитектура Fast-RCNN (<a href="https://arxiv.org/abs/1311.2524">Regions with CNN features</a>)
        2. FCOS (<a href="https://arxiv.org/abs/1904.01355">Fully Convolutional One-Stage Object Detection</a>)
    - Идентификация. Triplet Loss. ArcFace Loss.
    - Перенос стиля. Content Loss, Style Loss.
    - Генеративно - состязательные сети GAN
* Практика
    - Метрика Intersection over Union (IoU), ConvTranspose2d, Upsampling
    - Реализация сети UNet
    - Семантическая сегментация в PyTorch на датасетах NYUv2 и Pascal VOC
    - Реализация DCGAN (<a href="https://arxiv.org/abs/1511.06434">Deep Convolutional Generative Adversarial Networks</a>)
    - Генерация лиц на датасете Image Celeba
    - Обусловленный GAN на датасете MNIST
* Дополнительные материалы
    <a href="https://nanonets.com/blog/semantic-image-segmentation-2020/">A Complete guide to Semantic Segmentation</a>
</pre>
</details>

<details><summary> 6. Основы обработки естественного языка (NLP) </summary> 
<pre>
* Теория
    - Основные задачи NLP: Классификация, синтез и преобразование текста
    - Общий подход к решению NLP задач. Токены, эмбеддинги.
    - Векторизация текста
        1. Bag of Words (BOW)
        2. Word to Vector (Word2Vec)
    - Архитектуры: <a href="https://arxiv.org/abs/1408.5882">TextCNN</a>
* Практика
    - Задача регрессии: предсказание зарплаты по заголовку объявления на примере датасета <a href="https://www.kaggle.com/competitions/job-salary-prediction/data">Job Salary Prediction"</a>
        1. Токенизация nltk.tokenize.WordPunctTokenizer()
        2. Векторизация на word2vec-google-news-300
        3. Паддинг
        4. Векторизация категориальных фич: DictVectorizer
    - Обобщение TextCNN для текстовых и категориальных фич
        1. Текстовые фильтры со свертками разного размера nn.Conv1d
        2. Полносвязные слои для категориальных фич
    - Задача бинарной классификации отзывов на фильмы IMDB
* Дополнительные материалы
    Эмбеддинги <a href="https://radimrehurek.com/gensim/models/fasttext.html">fasttext</a>
</pre>
</details>

<details><summary> 7. Рекуррентные сети (RNN) и генерация текста </summary> 
<pre>
* Теория
    - Выделение именованных сущностей NER (Named-Entity Recognition)
    - Метрики NER. Типы и схемы подсчета ошибок 
    - Архитектура RNN
    - Генерация текста. Метрика Perplexity
    - Задача seq2seq. Энкодер и декодер
* Практика
    - Ручная реализация однослойной RNN
    - Генерация псевдонауыного текста по датасету <a href="https://www.kaggle.com/datasets/Cornell-University/arxiv">Arxiv</a>
    - Задача перевода текста с английского на русский язык.
* Дополнительные материалы
    Метрики генерации текста BLEU, ROUGE, METEOR, BERTScore <a href="https://habr.com/ru/articles/1002218/">на habr</a>
    ВВедение в метрику <a href="https://www.freecodecamp.org/news/what-is-rouge-and-how-it-works-for-evaluation-of-summaries-e059fb8ac840/">Rouge</a>. Гайд про <a href="https://dev.to/aws-builders/mastering-rouge-matrix-your-guide-to-large-language-model-evaluation-for-summarization-with-examples-jjg">Rouge</a> 
</pre>
</details>

<details><summary> 8. Обзор трансформера и сетей на его основе </summary> 
<pre>
* Теория
    - Механизм внимания (Attention). Вектора Query, Key, Value
    - Архитектура <a href="https://arxiv.org/abs/1706.03762">Трансформер</a> 
    - Positional encoding
    - Обзор моделей GPT - 1, 2, 3; Bert
* Практика
    - Обзор платформы <a href="https://huggingface.co/">HuggingFace </a> 
    - Библиотека transformers. transformers.pipeline, AutoModel, AutoTokenizer
    - Генерация текста открытыми моделями GPT-2, LLAMA, Mistral
    - Перевод Англ. - Рус. Helsinki-NLP/opus-mt-ru-en
    - Дообучение головы Bert для классификации отзывов imdb
</pre>
</details>

<details><summary> 9. Погружение в LLM </summary> 
<pre>
* Теория
    - Модели LLM 
    - Квантизация LLM. AWQ, GPTQ, bitsandbytes
* Практика
    - Решение NLP задач с помощью Gemma 2B
    - Построенме прототипа RAG системы, которая отвечает на вопросы пользователей по банку документов
    - Деплой LLM 
        1. Аренда облачного сервера
        2. Деплой LLM модели
        3. Деплой веб-интерфейса
* Дополнительные материалы
    - Про квантизацию <a href="https://huggingface.co/blog/4bit-transformers-bitsandbytes">4-bit and QLoRA</a> и <a href="https://kipp.ly/p/transformer-inference-arithmetic">Transformer Inference Arithmetic
    - <a href="https://docs.vllm.ai/en/stable/">VLLM</a>
    - Аргументы командной строки <a href="https://docs.vllm.ai/en/latest/serving/openai_compatible_server/#command-line-arguments-for-the-server">VLLM</a>
</a>
</pre>
</details>

#### Часть 2. CV
<details><summary> 1. Изображение и классические методы работы с ним </summary> 
<pre>
* Теория
    - Устройство матрицы камеры. Фильтр Байера. Дебайеризация
    - Цветовые модели RGB, HSV
    - Обработка изображений. Гамма-коррекция, эквализация гистограммы, CLAHE
    - Изменение размеров изображения Nearest neighbor, билинейная интерполяция, <a href=https://disk.yandex.ru/i/pxl9PLfrdDe8SQ">INTER_AREA</a>
    - Эффект алиасинга при downscaling
    - Бинаризация. Глобальный и адаптивный пороги. <a href="https://www.geeksforgeeks.org/python/otsu-thresholding-using-opencv/">Метод Оцу</a>
    - <a href="https://www.geeksforgeeks.org/python/python-opencv-morphological-operations/">Морфологические операции: эрозия, дилатация</a>
    - Выделение границ. Операцтор Собеля. Алгоритм Кэнни. Контуры, <a href="https://docs.opencv.org/4.x/dc/dcf/tutorial_js_contour_features.html">аппроксимация примитивами</a>
    - Гомография. Проективное преобразование
* Практика
    - Основы работы в OpenCV
        1. Работа с файлами, <a href="https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html">преобразование цветового формата изображения</a>
        2. Морфологические операции
        3. <a href="https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html">Выравнивание гистограмм</a> 
        4. Методы борьбы с алиасингом: размытие, интерполяция
        5. <a href="https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html">Выделение границ и контуров</a>
        6. Применение гомографии. <a href="https://www.geeksforgeeks.org/computer-vision/python-opencv-getrotationmatrix2d-function/">Поворот с маштабированием</a>
    - Задача обнаружения координат штрихкодов на изображении
* Дополнительные материалы
    <a href="https://habr.com/ru/articles/130300/">Проективная геометрия и основы стереозрения</a>
</a>
</pre>
</details>

<details><summary> 2. Классификация: постановка задачи, CNN, NAS-CNN </summary> 
<pre>
* Теория
    - Сверточные нейросети от AlexNet до семейства ResNets
    - MobileNetV1. Depthwise свертка. Pointwise свертка
    - MobileNetV2. Inverted Residual Block
    - <a href="https://docs.pytorch.org/vision/main/models/mobilenetv3.html">MobileNetV3</a>. Neural Architecture Search (NAS). Squeeze Excitation Block. Функция активации hswish
    - EfficientNets. Compound scaling. Fused MBConv
* Практика
    - Finetune моделей <a href="https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html">Resnet-50</a>, EfficientNet_b1
    - Задача классификации TinyImageNet
    - Реализация SE блока в pytorch
    - Inverted residual block. Linear bottleneck
    - Кастомная модификация MobileNetV3 в pytorch
   * Дополнительные материалы
    <a href="https://huggingface.co/papers/trending">Trending papers with code</a>
    <a href="https://docs.pytorch.org/vision/stable/models.html">Torchvision modelss</a>
    <a href="https://github.com/huggingface/pytorch-image-models">PyTorch Image Models</a>    
</a>
</pre>
</details>

<details><summary> 3. Как построить классный пайплайн обучения сети </summary> 
<pre>
* Теория
    - <a href="https://www.deeplearning.ai/ai-notes/initialization/index.html">Инициализация весов</a> по Хавьеру и Каймингу
    - CrossEntropy Loss, Focal Loss, Generalized Cross Entropy Loss
    - Adam, <a href="https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html">AdamW</a>, SGD с моментом
    - <a href="https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html">Autpomatic Mixed Precision (AMP)</a>. Autocast + GradScaler
    -Shedulers. StepLR, <a href="https://docs.pytorch.org/docs/2.11/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html">CosineAnnealingLR</a>. Планировщики с прогревом
    - Техники регуляризцации
        1. Weight decay
        2. <a href="https://docs.pytorch.org/docs/2.11/generated/torch.optim.swa_utils.AveragedModel.html">AveragedModel</a>. Экспоненциальное скользяцее среднее (EMA)
        3. Label smoothing
        4. Random Erasing
        5. Mixers. MixUp, CutMix
    - Аугментации
        1. AutoAugment
        2. RandAugment
        3. <a href="https://docs.pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html">TrivialAugment</a>
* Практика
    - Многоклассовая классификация с несбалансированными классами - распознавание дорожных знаков
        1. Аугментации
        2. Выравнивание сэмплирования. <a href="https://docs.pytorch.org/docs/2.12/data.html#torch.utils.data.WeightedRandomSampler">WeightedRandomSampler</a> 
    - Улучшение пайплайна обучения собственной кастомной модели MobileNet на TinyImageNet 
    - Реализация MixUp и MixCut аугментаций + модификация критерия кросс-энтропии
    - Запуск обучения в Kaggle с использованием <a href="https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html">Distributed Data Parallel (DDP)</a> для 2 GPU T4. CPU bottleneck!
    - Подбор lr шедулера
    - SGD vs Adamю. SGD победил
    - <a href="https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/">Stochastic Weight Averaging (SWA)</a>   
</a>
</pre>
</details>

<details><summary> 4. Классификация: VIT, CLIP </summary> 
<pre>
* Теория
    - Классический <a href="https://arxiv.org/abs/1706.03762">трансформер</a>:
        1. Энкодер и декодер
        2. Self, cross и masked self attention 
        3. Position encoding
        4. Матрицы Query, Key, Value
        5. Layer Norm 
    - <a href="https://research.google/blog/transformers-for-image-recognition-at-scale/">Vision transformer</a> <a href="https://arxiv.org/abs/2010.11929">(ViT)</a>:
        1. Линейная проекция векторизованных патчей
        2. Encoder block
        3. CLS - токен
        4. MLP head
    - <a href="https://arxiv.org/abs/2103.00020">Contrastive Language-Image Pre-training</a> (CLIP):
        1. Image Encoder
        2. Text Encoder
        3. Contrastive learning. Similaruty matrix
        4. Zero- и few-shot классификация
        5. Linear probing
* Практика
    - Реализация ViT в pytorch
    - Open-source имплементации CLIP - open-clip
    - Zero shot классификация Signs dataset
    - Zero shot на TunyImageNet
</a>
</pre>
</details>

<details><summary> 5. Сегментация </summary> 
<pre>
* Теория
    - Типы сегментации: семантическая, инстанс, паноптик. <a href-="https://arxiv.org/pdf/2001.05566">Обзор</a>
    - Метрики и функциии потерь: Жаккар (IoU), Дайс (Dice)
    - Архитектуры:
        1. <a href="https://arxiv.org/pdf/1505.04597">UNET</a>
        2. <a href="https://arxiv.org/pdf/1707.03718">LinkNet</a> ~ UNET(concat -> add))
        3. <a href="https://arxiv.org/pdf/1606.00915">DeepLab</a>. <a href="https://arxiv.org/pdf/1706.05587">DeeplabV3</a>
        4. Семейство Feature Pyramid Networks (<a href="https://arxiv.org/pdf/1612.03144">FPN</a>)
        5. <a href="https://arxiv.org/pdf/1908.07919">HRNet</a>
    - Segment Anything Model (<a href="https://arxiv.org/pdf/2304.02643">SAM</a>)
* Практика
    - Сегментация на датасете <a href="https://huggingface.co/datasets/sayakpaul/nyu_depth_v2">NYUv2</a>. Подбор подходящей архитектуры: UNET, UNET++, FPN
    - SAM - использование точек (кликов) и боксов как пользовательских промтов для zero-shot сегментации чего угодно
    - Реализация собственной версии UNET для решения задачи сегментации на Oxford Pet dataset:
        1. Уточнение масок при помощи SAM
        2. Адаптирование сверточного блока MBconv из <a href="https://arxiv.org/pdf/1801.04381">MobileNetV2</a>
        3. Реализация <a href="https://arxiv.org/pdf/1807.06521">CBAM: Convolutional Block Attention Module</a> - блока канального и пространственного внимания
        4. Реализация Res-блока с CBAM
        5. Обучение сети используя FocalLoss + DiceLoss
* Дополнительные материалы:
    <a href="https://d2l.ai/chapter_computer-vision/transposed-conv.html">Обратная свертка</a>. <a href="https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md">Анимация сверток</a>. <a href="https://arxiv.org/pdf/1511.07122v3">Dilated/Atrous свертки</a>. 
    <a href="https://arxiv.org/pdf/1901.02446">Panoptic FPN</a>
</a>
</pre>
</details>

<details><summary> 6. Детекция: метрики, two-stage, one-stage </summary> 
<pre>
* Теория
    - Типы боксов детектирования: Horizontal Bounding Box (HBB), Oriented Bounding Box (OBB), Polygon
    - Датасеты: Pascal VOC, MS COCO, ILSVRC, Objects365, Open ImagesDataset (OID) 
    - Метрики: IoU, mAP - mean Average Precision. AP_50, AP_75, AP (mean by IoU 50:5:95)
    - Two-stage детекторы: R-CNN -> Fast R-CNN -> Faster R-CNN -> <a href="https://arxiv.org/abs/1703.06870">Mask R-CNN</a>
    - Особенности архитектуры two-stage:
        1. Генерация боксов - кандидатов RoI=Regions of Interest  (by Selective search/RPN anchors)
        2. Non Maximum Suppression (NMS) - отфильтровываем похожие боксы по IoU
        3. RoI Pooling / RoI Allign - проекция боксов на Feature Map
        4. Классификация боксов + уточнение их координат + instance сегментация (только в Mask R-CNN)
* Практика
    - Transfer learning Mask R-CNN на датасете <a href="https://www.cis.upenn.edu/%7Ejshi/ped_html/">Penn-Fudan</a>
    - Ручная реализация метрики mAP
</pre>
</details>

<details><summary> 7. Детекция: one-stage, anchor-free детекторы, YOLO </summary> 
<pre>
* Теория
    - One-stage детекторы: 
        1. <a href="https://arxiv.org/pdf/1506.02640">YOLO v1</a> 448x448 -> 7x7x30 - предсказание 20 классов, по 2 объекта в каждой ячейке финального тензора (всего 98 объектов)
        2. <a href="https://arxiv.org/pdf/1512.02325">SSD</a> (single shot detector) - предсказания с фичемап разного размера 
        3. FPN (feature pyramid network) - переагрегация фичемап разного размера
        4. <a href="https://arxiv.org/pdf/1708.02002">RetinaNet</a> - FPN like + FocalLoss
    - Anchor-free детекторы:
        1. <a href="https://arxiv.org/pdf/1904.07850">CenterNet</a> - Objects as points, предсказание центра каждого объекта и размера бокса, без якорей и NMS
        2. <a href="https://arxiv.org/pdf/1904.01355">FCOS</a> - Fully Convolutional One-Stage Object Detection - классификация + предсказание центральности + регрессия на ширину и высоту объекта
        3. <a href="https://arxiv.org/pdf/2005.12872">DETR</a> - detection transformer - первый трансформер для детекции. Энкодер кодирует части изображения. Декодер принимает object queries и предсказывют для них "объект/не объект" при помощи венгерского алгоритма. 
    - <a href="https://arxiv.org/pdf/2304.00501v1">YOLO family</a>:
        1. v1-v3 (2015-2018) Original YOLO by Joseph Redmon
        2. <a href="https://arxiv.org/pdf/2004.10934">v4</a> (2020) by Alexey Bochkovskiy et al.
        3. v5+ (2020 - ...) by <a href="https://github.com/ultralytics/ultralytics">Ultralytics</a> 
    - Портирование сетей для инференса: ONNX, TensorRT, OpenVINO
* Практика
    - Знакомство с проектом Ultralytics
    - Конфигурации моделей и процесса обучения
    - Механизм scale и multi-scale
    - Механизм mosaic и close mosaic
    - Распределение весов у лоссов во время обучения
    - Критерии качества модели, fitness-функция
* Дополнительные материалы
    <a href="https://arxiv.org/pdf/1905.05055">Object Detection in 20 Years: A Survey</a>
    <a href="https://blog.roboflow.com/guide-to-yolo-models/">Особенности сетей YOLO</a>
    <a href="https://docs.ultralytics.com/ru/integrations/onnx">Экспорт YOLO в ONNX</a>
    <a href="https://docs.openvino.ai/2024/index.html">Дока OpenVINO</a>
    <a href="https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/">Speeding Up Deep Learning Inference Using TensorFlow, ONNX, and NVIDIA TensorRT</a>
</pre>
</details>

<details><summary> 8. Face recognition/ ReID/ Image retrieval </summary> 
<pre>
* Теория
    - Постановка задачи, Open set vs Closed set problem. Верификация (1 v 1) и идентификация (1 v N).
    - Метрики: TPIR at FPIR=10^(-N) - какую долю правильных идентификаций (TPIR) мы получаем при строго заданном уровне ложных срабатываний (FPIR)
    - Функции потерь Mertic Learning (группировка на гиперсфере):
        1. Contrastive Loss
        2. Triplet Loss
        3. Margin-based Losses: ArcFace
    - Knowledge Distillation - передача "навыка" от тяжелой сети (учителя) более легкой (ученику). Логиты сети учителя смягчаются Softmax с температурой и являются таргетами для ученика.
    - Re-Identification - image query -> поиск того же самого объекта с других ракурсов
    - Image retrieval - image query -> список похожих изображений в базе
* Практика
    - Распознвавние лиц с помощью InsightFace - датасет <a href ="https://www.kaggle.com/datasets/amiralikalbasi/images-of-friends-character-for-face-recognition/data">Friends</a>
</pre>
</details>

<details><summary> 9. Трекинг </summary> 
<pre>
* Теория
    - Multiple Object Tracking (MOT). Метрики:
        1. MOT Accuracy. MOTA = 1 - (FN + FP + IDSW)/gtDet. MOTA не учитывает True Positives, не отражает фрагментацию и не оценивает точность локализации предсказанных боксов относительно ground truth.
        2. MOT Precision. MOTP = Sum(Overlap(pred, gt))/Matches. 
        3. IDF1 - F1 для детекции
        4. HOTA (Higher Order Tracking Accuracy). Учитывает как качество детекции, так и качество ассоциации.
    - Simple Online and Real-Time Tracker (<a href="https://arxiv.org/pdf/1602.00763">SORT</a>) - быстрый и простой треккер, решает проблемы неудачной локализации, FP и FN, но не уменьшает ID switches. Шаги:
        1. Детекция
        2. <a href="https://habr.com/ru/companies/singularis/articles/516798/">Фильтр Калмана</a> - коррекция данных с детектора
        3. Ассоциация - венгерский алгоритм
    - SORT - type трекеры:
        1. DeepSort - SORT + отдельная простая CNN для ReID
        2. FairMOT - Детектор anchor-free как в Centernet + голова для дескриптора
        3. ByteTrack - Продление трека для боксов с низким IoU
        4. BoT SORT- учет движения камеры.
    - Single Object Tracking (SOT). Особенности:
        1. Детектор работает только на первом кадре, на последующих ищется объект в окрестности бокса с предыдущего кадра. Для этого часто используются симские сети, но есть и классические методы OpenCV
        2. Нет необходимаости использовать IoU, так как нет ассоциации между разными обхектами
* Практика
    - SORT изнутри: 
        1. Фильтр Калмана
        2. Фаза ассоциации
    - DeepSORT tracker на MOT15 challenge - трекинг людей в торговом центре. Подсчет количества пересечений линии.

</pre>
</details>

<details><summary> 10. SSL & VLMs </summary> 
<pre>
* Теория
    - Self-supervised learning (SSL). Классические методы:
        1. Предсказание контекста, разгадывание пазла (jigsaw), колоризация изображений, восстановление маскированных частей,  предсказание поворота
    - Архитектуры сиамских подходов к SSL для CNN:
        1. SimCLR (Simple Contrastive Learning of visual Representations) - использование Contrastive Loss для пар двух аугментаций разных картинок
        2. BYOL (Bootstrap Your Own Latent) - Веса сети - учителя обновляются через EMA весов сети - ученика. Используются только позитивные кропы
        3. DINO (DIstillation witn NO labels) - похож на BYOL, но другой подход к созданию views, центрирование чтобы избежать feature collapse, Softmax с температурой + CE
    - Трансформерные методы для SSL:
        1. Masked AutoEncoders (MAE) - изображение нарезается на патчи и часть их маскируется
        2. DINOv2 - сочетает идеи первого DINO и MAE. Для сети - учителя изображение показывают целиком, а для сети- ученика часть патчей маскируется
    - Vision Language Models (VLM) - мультимодальные модели, работающие как с текстом, так и с изображениями.
    - Grounding DINO (DETR with Improved de Noising anchOr - другой динозавр, не SSL) - модель для open-set-детекции в различных постановках 
        1. По тексовому описанию
        2. По отдельным тегам
* Практика
    - Мультимодальные задачи в zero-shot постановке
        1. Генерация описаний изображений с помощью BLIP
        2. Фильтрация результатов с помощью CLIP
        3. Генерация тегов изображений с помощью RAM
        4. Определение Bbox-ов с помощью Grounding Dino
        5. Сегментация с помощью Grounded SAM
</pre>
</details>

<details><summary> 11. OCR </summary> 
<pre>
* Теория
    - Постановка задачи Optical Charachter Recognition. Печатные тексты, "in the wild". 
    - Метрики на основе расстояния Левенштейна - CER (Character Error Rate), WER (Word Error Rate)
    - Пайплайн OCR:
        1. Детекция текста
        2. Распознавание текста
        3. Извлечение информации
    - Регрессионные методы детекции (якорные боксы для правильных текстов). Примеры: CTPN, EAST, RSDD.
    - Сегментационные методы детекции (маски для букв и слов + постобработка). Примеры: CRAFT, DBNet, DBNet++, PSENet, PANet, TextSnake.
    - CRAFT (Chatacter Awareness For Text detection) - предсказывает 2D-гауссианы для центров символов и связей (affinity) между ними, обучаясь итеративно на псевдоразметке, полученной из word-level боксов реальных данных после предобучения на синтетике.
    - DBNET (Differentiable Binarization Network) - предсказывает одновременно probability map и threshold map, используя дифференцируемую аппроксимацию бинаризации, что позволяет обучать порог end-to-end и избегать ручного подбора в постпроцессинге
    - Типы распознавания текста:
        1. CTC-based — для Regular Text Recognition (горизонтальный текст): детекция + CNN-экстрактор + RNN-транскриптор с CTC-loss, предсказание на уровне слов/строк. Пример: CRNN
        2. Attention-based — для Irregular Text Recognition (кривой/произвольный текст): сегментация + seq2seq с механизмом внимания, предсказание посимвольно или n-граммами. Пример: SAR
    - Key Information Extraction (KIE) - извлечение структуры текста в виде словаря {key: value}.
        1. SDMGR — графовая модель: вершины графа = эмбеддинги слов (визуальные из U-Net + текстовые из Bi-LSTM), рёбра = пространственные связи; предсказывает классы слов (название, цена, итог) -> структурированный JSON.
        2. LayoutLM — языковая модель для документов: объединяет визуальные эмбеддинги (из детектора боксов) + текстовые эмбеддинги (из OCR + предобученной LM) + позиционные эмбеддинги, затем пропускает через трансформер для классификации слов.
        3. DONUT — end-to-end трансформер: на вход — изображение документа, на выход — JSON
    - Подходы к OCR:
        1. Multi-stage (Tesseract, EasyOCR, PaddleOCR, MMOCR) — конвейер "детекция -> распознавание -> KIE", легковесные, сохраняют конфиденциальность
        2. End-to-end (MMOCR, DONUT) — единая модель от изображения до структурированного результата, проще в использовании, но ресурсоёмкие.
* Практика
    - Распознавание штрихкодов
    - Реализация собственной CRNN на базе ResNet50 и GRU
</pre>
</details>