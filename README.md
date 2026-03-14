### Обучение на курсе Karpov Cources [DL engineer: CV](https://karpov.courses/deep-learning). Старт 02.02.2026.
### Путевые заметки веду в [моем блоге в ТГ](https://t.me/dl_journey)
#### Актуальный прогресс:
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


