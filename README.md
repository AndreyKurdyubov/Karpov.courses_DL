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
    - Архитектуры сверточных сетей. Lenet VGG. Inception. ResNet
* Практика
    - Свертка, pooling
    - Разбиение на батчи. Dataset и Dataloader
    - Аугментация. RandomResizedCrop. Rotate. Hotizontal flip. Resize. Normalize
    - Построение сверточной сети для классификации объектов из датасета CIFAR10
    - Finetuning готовой модели на примере ResNet, mobilnet
    - Исследование косинусной схожести эмбедингов объектов одного класса
</pre>
</details>


