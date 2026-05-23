### Курс [DL engineer: CV](https://karpov.courses/deep-learning) от Karpov Cources
### Мой [блог в ТГ](https://t.me/dl_journey)
### Старт прохождения 02.02.2026. Актуальный прогресс на 26.04.2026:
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
    - Классический <a href="https://arxiv.org/abs/1706.03762">трансформер</a> :
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