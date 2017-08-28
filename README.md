# Практика. Классификация изображений с использованием возможностей модуля dnn библиотеки OpenCV

## Содержание

1. [Цели][purpose]
1. [Задачи][tasks]
1. [Последовательность выполнения работы][scheme]
1. [Структура репозитория][repo-structure]
1. [Полезные ссылки][refs]

## Цели

**Цель** данной работы состоит в том, чтобы изучить некоторые
возможности модуля dnn библиотеки компьютерного зрения
[OpenCV][opencv]. В частности, изучить методы организации
работы с моделями глубоких нейросетей в формате библиотеки
[Caffe][caffe] и их применения для решения задачи классификации
изображений.

## Задачи

**Основные задачи**

Основная задача состоит в том, чтобы разработать консольное
приложение для классификации изображений с использованием
обученных моделей глубоких нейронных сетей. Приложение должно
обеспечивать следующий функционал:

1. Загрузка изображения в формате RGB для последующей классификации.
1. Загрузка и чтение модели глубокой сети, хранящейся в формате
   библиотеки [Caffe][caffe]. Для определенности предлагается
   использовать модель GoogLeNet.
1. Предварительная обработка изображения для подачи на вход
   классификатору:
   
   - Выделение области изображения, размер которой соответствуют
     размеру входа нейронной сети. Сеть GoogLeNet принимает на вход
	 изображение размера 224x224 в формате RGB.
   - Вычитание среднего значения интенсивности, полученного
     на тренировочной выборке. Для сети GoogLeNet среднее значение
	 составляет (104, 117, 123).
   
1. Определение класса, которому принадлежит изображение.
   На выходе сети формируется вектор достоверностей принадлежности
   изображения каждому из возможных классов, поэтому необходимо
   выбрать класс с максимальной достоверностью.

**Дополнительные задачи**

Дополнительно в разработанное приложение можно добавить
интерфейсную часть:

1. Отображение исходного изображения.
1. Отображение области изображения, выбранной для классификации,
   и названия класса, которому оно принадлежит с наибольшей
   вероятностью.
1. Добавить предварительную обработку изображения посредством
   применения различных методов сглаживания и фильтрации.
   Проанализировать, как изменится результат классификации
   и значение достоверности принадлежности классу.

Примечание: для повышения сложности дополнительных задач
можно выполнить отображение в одном окне.

## Программное обеспечение

Для выполнения практической работы необходимо использовать
следующее программное обеспечение:

1. Утилита [CMake][cmake] (версия не ниже 2.8).
1. Git-клиент (например, [git-scm][git-scm]).
1. Библиотека [OpenCV][opencv] версии 3.3.0.
1. Среда разработки (далее используется
   [Microsoft Visual Studio 2015 Win64][ms-vs2015], поскольку
   в состав пакета OpenCV 3.3.0 входят бинарники, собранные
   для указанной версии среды).

## Структура репозитория

Репозиторий содержит следующие директории и файлы:

- `samples` - директория, содержащая шаблонный файл исходного кода
  приложения.
- `README.md` - настоящее описание.
- `CMakeLists.txt` - основной файл для сборки проекта с помощью CMake.
- `help` - директория, содержащая описание и примеры использования
  основных функций библиотеки OpenCV, которые потребуются для
  выполнения практической работы.
- `.gitignore` - перечень директорий/файлов, которые игнорируются
  системой контроля версий.

## Общая последовательность выполнения работы

1. Загрузить проект из репозитория GitHub и создать рабочую ветку
   в соответствии с вашим ФИО (например, IvanovAA).

   ```bash
   git clone https://github.com/UNN-VMK-Software/opencv-dnn-practice
   cd opencv-dnn-practice
   git checkout -b IvanovAA
   cd ..
   ```
   
   Примечание: для более детального изучения возможностей системы
   контроля версий Git рекомендуется воспользоваться
   [материалами Летней межвузовской школы 2016][itseez-ss-2016-practice-1].

1. Создать директорию для размещения файлов решения и проекта,
   перейти в эту директорию и собрать файлы проекта с помощью CMake.

   ```bash
   mkdir opencv-dnn-practice-build
   cd opencv-dnn-practice-build
   cmake -DOpenCV_DIR="c:\Program Files\OpenCV-3.3.0\opencv\build" -G "Visual Studio 14 2015 Win64" ..\opencv-dnn-practice
   ```

   Примечания: В опции `OpenCV_DIR="c:\Program Files\OpenCV-3.3.0\opencv\build"`
   необходимо указать путь до 	файла `OpenCVConfig.cmake` библиотеки
   OpenCV-3.3.0.

1. Открыть решение `opencv-dnn-practice-build/opencv_dnn_practice.sln`
   и собрать проекты, входящие в его состав, нажав правой кнопкой мыши
   по проекту `ALL_BUILD` и выбрав из выпадающего меню команду `Rebuild`.
   Бинарные файлы будут скопированы в директорию `opencv-dnn-practice-build/bin`.
   Примечание: перед компиляцией при необходимости следует выбрать
   режим сборки `Debug` (по умолчанию) или `Release`.

1. Запустить пустое приложение. Для этого необходимо открыть
   консоль из директории `opencv-dnn-practice-build` и выполнить
   `dnn_sample.exe`. При запуске возможно возникновение ошибки вида
   "Запуск программы невозможен, так как на компьютере отсутствует opencv_world330.dll".
   Это означает, что при запуске не были обнаружены бинарные файлы
   библиотеки OpenCV. Чтобы решить указанную проблему, можно
   прописать в переменную окружения `PATH` путь до библиотек
   `C:\Program Files\OpenCV-3.3.0\opencv\build\x64\vc14\bin`,
   либо скопировать соответствующий файл библиотеки к бинарному
   файлу приложения. Примечание: обратите внимание, что у вас
   путь может отличаться.

1. Открыть файл исходного кода `dnn_sample.cpp` в проекте `dnn_sample`
   для дальнейшего решения задач практической работы.

1. Добавить код, обеспечивающий чтение параметров командной строки.
   Приложение должно принимать следующие параметры:

   - Файл с изображением.
   - Текстовый файл с описанием модели глубокой сети в формате
     `prototxt`, который обрабатывается библиотекой [Caffe][caffe]
	 (сеть GoogLeNet и некоторые другие есть в пакете OpenCV
	 `c:\Program Files\OpenCV-3.3.0\opencv\sources\samples\data\dnn\bvlc_googlenet.prototxt`).
   - Бинарный файл `*.caffemodel` с параметрами обученной модели
     глубокой сети. Обученную сеть GoogLeNet можно скачать
	 по [ссылке][caffemodel].
   - Файл, содержащий перечень распознаваемых классов изображений.
     Для набора ImageNet указанный файл также есть в пакете OpenCV
	 `c:\Program Files\OpenCV-3.3.0\opencv\sources\samples\data\dnn\synset_words.txt`.
   
   Примечание: обратите внимание, что у вас пути могут отличаться.
   
   Указание: чтение параметров рекомендуется реализовать
   с использованием класса [`cv::CommandLineParser`][commline-parser].

1. Добавить код для загрузки изображения для классификации.

1. Добавить код для загрузки и чтения модели глубокой
   нейронной сети.

1. Добавить код для выполнения предварительной
   обработки изображения.

1. Добавить код для установки полученного изображения
   в качестве входа сети.

1. Добавить код для выполнения прямого прохода сети для заданного входа
   и получения вектора достоверностей принадлежности каждому классу.

1. Добавить код для определения класса изображения - класса,
   для которого получено максимальное значение достоверности.
   Примечание: для этого необходимо в векторе достоверностей
   найти индекс максимального элемента, загрузить перечень
   классов изображений из файла и получить содержательное
   описание класса.

1. Выложить результаты решения основных задач на GitHub
   в созданную вами ветку.

   ```bash
   git commit -m "Main problems" -a
   git push origin IvanovAA
   ```

1. Выполнить дополнительные задачи практической работы.


<!-- LINKS -->

[purpose]: https://github.com/UNN-VMK-Software/opencv-dnn-practice#%D0%A6%D0%B5%D0%BB%D0%B8
[tasks]: https://github.com/UNN-VMK-Software/opencv-dnn-practice#%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B8
[scheme]: https://github.com/UNN-VMK-Software/opencv-dnn-practice#%D0%9F%D0%BE%D1%81%D0%BB%D0%B5%D0%B4%D0%BE%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D0%BE%D1%81%D1%82%D1%8C-%D0%B2%D1%8B%D0%BF%D0%BE%D0%BB%D0%BD%D0%B5%D0%BD%D0%B8%D1%8F-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B
[repo-structure]: https://github.com/UNN-VMK-Software/opencv-dnn-practice#%D0%A1%D1%82%D1%80%D1%83%D0%BA%D1%82%D1%83%D1%80%D0%B0-%D1%80%D0%B5%D0%BF%D0%BE%D0%B7%D0%B8%D1%82%D0%BE%D1%80%D0%B8%D1%8F
[refs]: https://github.com/UNN-VMK-Software/opencv-dnn-practice#%D0%9F%D0%BE%D0%BB%D0%B5%D0%B7%D0%BD%D1%8B%D0%B5-%D1%81%D1%81%D1%8B%D0%BB%D0%BA%D0%B8
[opencv]: http://opencv.org
[caffe]: http://caffe.berkeleyvision.org
[caffemodel]: http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
[itseez-ss-2016-practice-1]: https://github.com/itseez-academy/itseez-ss-2016-practice/blob/master/docs/README_1.md
[cmake]: http://cmake.org
[git-scm]: https://git-scm.com
[ms-vs2015]: https://www.visualstudio.com/ru/vs/older-downloads
[commline-parser]: http://docs.opencv.org/trunk/d0/d2e/classcv_1_1CommandLineParser.html#details
