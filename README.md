# 🍓 Berry Classifier – Микросервисный ML-проект с Telegram-ботом

> Production-ready система классификации ягод с современными ML-подходами, CI/CD, микросервисной архитектурой и интеграцией в Telegram.

---
### 💡Итоговая модель на тестовой выборке показала точность: **0.8420**

## Используемые в проекте подходы

- 🔍 **Transfer Learning:** Модель построена на EfficientNet-B1 с предобученными весами (ImageNet).
- 🔁 **Fine-Tuning:** Дообучение на собственных данных для 44 классов ягод.
- 🧪 **Grid Search:** Подбор гиперпараметров (lr, batch size, augmentations).
- 🧘 **Anti-overfitting:** Использован EarlyStopping, регуляризация и аугментации.
- 📦 **ONNX Export:** Модель экспортирована и загружается через `onnxruntime` в FastAPI.
- 💡 **Гибкость:** Модель легко переобучить, изменив параметры и веса — все конфиги централизованы.
- 📊 **PostgreSQL** для хранения результатов инференса и оригинальных изображений.
- 📋 **Structured Logging:** Логи сохраняются в volume, читаемы и централизованы.
- ⚙ **Docker Compose:** Все сервисы подняты отдельно — API, бот, БД.
- 🚀 **CI/CD:** GitHub Actions деплоит проект на VPS по push в `master`.

---

## ⚙️ Используемый стек

| Категория           | Технологии                                                                 |
|---------------------|----------------------------------------------------------------------------|
| Язык                | Python 3.12                                                                |
| Модель              | EfficientNet-B1 (transfer learning, ONNX export)                           |
| Инференс            | ONNXRuntime, Pillow, Torchvision                                           |
| Telegram Bot        | aiogram 3.x (async/await, FSM, router system)                              |
| API                 | FastAPI (async, type hints, Pydantic)                                      |
| База данных         | PostgreSQL                                                                 |
| Архитектура         | Микросервисная, через Docker Compose                                       |
| Логирование         | YAML config, `logging`, сохранение логов в volume                         |
| CI/CD               | GitHub Actions → VPS (`rsync`, `docker-compose`, `prune`)                 |
| Хранилище модели    | `.onnx` модель в `/models/`                                                |
| Отправка запросов   | `aiohttp` для Telegram, `requests` к API                                   |
| Хранение данных     | изображения и предсказания в БД                                            |
| Мониторинг          | логирование всех событий, включая ошибки и успешные инференсы             |

---

## 🤖 Telegram-бот

<img src="https://github.com/user-attachments/assets/c13e350b-3607-4dfd-9c1d-408643368630" width="70%" />

### Ссылка: @berry_classifier_bot

---

## 🧠 Архитектура проекта
User  
↓  
Telegram Bot (aiogram)  
↓  
FastAPI (/predict) ← ONNX Inference  
↓  
PostgreSQL (image + prediction)

    

- **Бот:** получает изображение, отправляет на API.
- **API:** обрабатывает изображение, проводит инференс, сохраняет данные в БД.
- **Сервисы:** изолированы в контейнерах, общаются по сети через docker-compose.

---



## 🗃️ Структура проекта
```text
Berry_Classifier/  
├── .github/  
│   └── workflows/  
│       └── deploy.yml           # CI/CD pipeline (например, GitHub Actions для деплоя Docker-контейнеров)  
│  
├── data/                        # Все, что связано с датасетом  
│   ├── raw/                     # Оригинальный датасет (262 класса фруктов и ягод)  
│   ├── processed/               # Предобработанные изображения  
│   │   ├── resized_Berries_Fruit-262/   # Преобразованные изображения (resize, split)  
│   │   │   ├── train/           # Тренировочная выборка  
│   │   │   ├── val/             # Валидационная выборка  
│   │   │   └── test/            # Тестовая выборка  
│   │   └── split_Berries_Fruit-262/     # Разделение с фиксированным random seed  
│  
├── logs/                        # Хранение логов работы моделей, API и бота  
│  
├── models/                      # Сохраненные обученные модели и скрипты экспорта  
│   ├── efficientnetB1_best_model.onnx        # Экспортированная ONNX-модель  
│   └── export_to_onnx.py                # Скрипт конвертации PyTorch → ONNX  
│  
├── services/                    # Микросервисы  
│   ├── telegram_bot/            # Telegram-бот на aiogram  
│   │   ├── handlers.py  
│   │   └── main.py  
│   ├── inference_api/          # REST API на FastAPI, работающий с ONNX  
│   │   ├── main.py  
│   │   └── utils.py  
│   ├── database/               # Работа с PostgreSQL  
│   │   ├── models.py           # SQLAlchemy ORM  
│   │   ├── crud.py             # CRUD-операции  
│   │   └── connection.py       # Подключение к БД  
│   ├── Dockerfile.api          # Dockerfile для API  
│   ├── Dockerfile.bot          # Dockerfile для бота  
│   └── docker-compose.yml      # Компоновка всех сервисов в одну сеть  
│  
├── src/                         # Исходный ML-код  
│   ├── configs/                # Гибкие конфигурации обучения и инференса  
│   │   └── config.yaml  
│   ├── data_loaders/           # Кастомные DataLoader'ы (torch)  
│   │   └── data_loader.py  
│   ├── models/                 # Архитектура моделей (EfficientNet и др.)  
│   │   └── efficientnetB1.py  
│   ├── training/               # Логика обучения  
│   │   └── trainer.py  
│   │   └── train.py  
│   └── inference/              # Скрипты для инференса  
│       └── inference.py  
│  
├── logging_config.yaml         # Единая конфигурация логгирования   
├── README.md                   # Документация проекта  
└── .env                        # Переменные окружения   
```
---

## 🪄 Этапы запуска проекта
### Подготовка модели:

- Обучение модели EfficientNetB1 с использованием transfer learning

- Подбор гиперпараметров через Grid Search

- Применение аугментаций, регуляризации, EarlyStopping

- Экспорт модели в ONNX через torch.onnx.export

### Разделение на микросервисы:

- API (FastAPI + ONNXRuntime)

- Telegram-бот (aiogram 3)

- PostgreSQL (через volume)

### Контейнеризация:

- Каждый сервис в отдельном Docker-контейнере

- Связаны через docker-compose

- Логи сохраняются в общие volume'ы

### CI/CD:

- GitHub Actions деплоит на сервер

- Автоматическая очистка старых файлов

- Сборка, запуск и обновление контейнеров

---

## 👨‍💻 Автор  
## Касимов Бахтияр  
- Data Scientist / ML Engineer / Python Backend Developer   

<img src="https://github.com/user-attachments/assets/a03a04d9-5fb6-42bb-a251-ace15ed2ac63" width="70%" />

### 📧 bkasimov123@gmail.com
