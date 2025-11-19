# 📚 Индекс документации

Полный список всей документации проекта Banking Analysis Suite.

## 🎯 Начало работы

### Для пользователей:
1. **[README.md](README.md)** - Главная страница проекта
2. **[api/START_HERE.md](api/START_HERE.md)** - Быстрый старт API
3. **[api/QUICKSTART_UNIFIED.md](api/QUICKSTART_UNIFIED.md)** - Быстрый старт Unified API

### Для разработчиков:
1. **[CONTRIBUTING.md](CONTRIBUTING.md)** - Как внести вклад
2. **[GITHUB_SETUP.md](GITHUB_SETUP.md)** - Публикация на GitHub
3. **[DEPLOYMENT.md](DEPLOYMENT.md)** - Развертывание на production

---

## 📁 Документация по модулям

### 🌐 API (api/)
```
api/
├── README.md                    - Обзор API модуля
├── START_HERE.md                - Точка входа для новых пользователей
├── QUICKSTART_UNIFIED.md        - Быстрый старт Unified API
├── UNIFIED_API_README.md        - Полная документация Unified API
├── API_USAGE.md                 - Документация старого OperCode API
└── INTEGRATION_SUMMARY.md       - Техническая сводка интеграции
```

**Что читать:**
- Новичкам: `START_HERE.md`
- Для использования API: `UNIFIED_API_README.md`
- Для разработки: `INTEGRATION_SUMMARY.md`

### 🤖 Fine-tuning (gemma_finetuning/)
```
gemma_finetuning/
├── README.md                    - Обзор модуля обучения
├── PROJECT_OVERVIEW.md          - Обзор проекта
├── MONITORING_GUIDE.md          - Мониторинг обучения
├── CLASS_BALANCING.md           - Балансировка классов
├── IMPROVEMENTS_SUMMARY.txt     - Сводка улучшений
├── FINAL_SUMMARY.txt            - Финальная сводка
└── QUICK_CHECK.txt              - Быстрая проверка
```

**Что читать:**
- Для обучения модели: `README.md`
- Для мониторинга: `MONITORING_GUIDE.md`
- Для оптимизации: `CLASS_BALANCING.md`

### 👤 NER (NER/)
```
NER/
└── README.md                    - Документация NER модуля
```

**Что читать:**
- Для использования NER: `README.md`

### 📊 Data (data/)
```
data/
├── purpose_codes.txt            - Словарь кодов операций
├── INFOGRAPHIC_SUMMARY.txt      - Инфографическая сводка
└── INDEX.txt                    - Индекс данных
```

---

## 🗂️ Документация по темам

### 🚀 Установка и запуск

| Документ | Описание | Для кого |
|----------|----------|----------|
| [README.md](README.md) | Основная установка | Все |
| [api/START_HERE.md](api/START_HERE.md) | Быстрый старт API | Пользователи |
| [api/QUICKSTART_UNIFIED.md](api/QUICKSTART_UNIFIED.md) | Unified API | Пользователи |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production деплой | DevOps |

### 📖 API документация

| Документ | Описание | Тип API |
|----------|----------|---------|
| [api/UNIFIED_API_README.md](api/UNIFIED_API_README.md) | Полная документация | Unified (OperCode + NER) |
| [api/API_USAGE.md](api/API_USAGE.md) | Старый API | OperCode только |
| [api/INTEGRATION_SUMMARY.md](api/INTEGRATION_SUMMARY.md) | Техническая сводка | Оба |

### 🧠 Machine Learning

| Документ | Описание | Тема |
|----------|----------|------|
| [gemma_finetuning/README.md](gemma_finetuning/README.md) | Обучение модели | Fine-tuning |
| [gemma_finetuning/MONITORING_GUIDE.md](gemma_finetuning/MONITORING_GUIDE.md) | Мониторинг | Training |
| [gemma_finetuning/CLASS_BALANCING.md](gemma_finetuning/CLASS_BALANCING.md) | Балансировка | Optimization |
| [NER/README.md](NER/README.md) | NER система | Named Entity Recognition |

### 🛠️ Разработка

| Документ | Описание | Для кого |
|----------|----------|----------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Гайд контрибьютора | Разработчики |
| [GITHUB_SETUP.md](GITHUB_SETUP.md) | Публикация на GitHub | Мейнтейнеры |
| [requirements.txt](requirements.txt) | Зависимости | Все |
| [.gitignore](.gitignore) | Игнорируемые файлы | Разработчики |

---

## 🎓 Обучающие материалы

### Для начинающих:
1. Прочитайте [README.md](README.md)
2. Следуйте [api/START_HERE.md](api/START_HERE.md)
3. Изучите примеры в [api/UNIFIED_API_README.md](api/UNIFIED_API_README.md)

### Для продвинутых:
1. Изучите [gemma_finetuning/README.md](gemma_finetuning/README.md)
2. Настройте мониторинг [gemma_finetuning/MONITORING_GUIDE.md](gemma_finetuning/MONITORING_GUIDE.md)
3. Оптимизируйте [gemma_finetuning/CLASS_BALANCING.md](gemma_finetuning/CLASS_BALANCING.md)

### Для DevOps:
1. Следуйте [DEPLOYMENT.md](DEPLOYMENT.md)
2. Настройте мониторинг production
3. Настройте backup и recovery

---

## 📊 Статистика документации

```
Всего документов: 20+
Общий объем: ~150 KB
Языки: Русский, English (в коде)

Категории:
├── Быстрый старт: 3 документа
├── API: 5 документов
├── Fine-tuning: 6 документов
├── Разработка: 3 документа
└── Deployment: 1 документ
```

---

## 🔍 Поиск по документации

### По задачам:

**"Как запустить API?"**
→ [api/START_HERE.md](api/START_HERE.md)

**"Как обучить модель?"**
→ [gemma_finetuning/README.md](gemma_finetuning/README.md)

**"Как внести вклад?"**
→ [CONTRIBUTING.md](CONTRIBUTING.md)

**"Как задеплоить на сервер?"**
→ [DEPLOYMENT.md](DEPLOYMENT.md)

**"Как использовать NER?"**
→ [NER/README.md](NER/README.md)

**"Как опубликовать на GitHub?"**
→ [GITHUB_SETUP.md](GITHUB_SETUP.md)

---

## 📝 Обновление документации

При добавлении новой документации:

1. Добавьте файл в соответствующую папку
2. Обновите этот индекс
3. Добавьте ссылки в README.md
4. Обновите CONTRIBUTING.md если нужно

---

## 💡 Полезные ссылки

### Внутренние:
- [Главная страница](README.md)
- [API документация](api/README.md)
- [Fine-tuning гайд](gemma_finetuning/README.md)

### Внешние:
- [Hugging Face - Gemma](https://huggingface.co/google/gemma-2-9b-it)
- [Ollama - Qwen](https://ollama.com/library/qwen2.5)
- [FastAPI документация](https://fastapi.tiangolo.com/)
- [Transformers документация](https://huggingface.co/docs/transformers)

---

**Документация полная и актуальная!** ✅

Последнее обновление: 2024-11-19

