# 📦 Подготовка проекта для GitHub

## 🚀 Быстрая публикация

### 1. Инициализация Git репозитория

```bash
cd /home/user/Desktop/RPA

# Инициализация
git init

# Добавление файлов
git add .

# Первый commit
git commit -m "Initial commit: Banking Analysis Suite"
```

### 2. Создание репозитория на GitHub

1. Перейдите на https://github.com/new
2. Создайте новый репозиторий с именем `RPA` или `banking-analysis-suite`
3. **НЕ** добавляйте README, .gitignore или LICENSE (они уже есть)

### 3. Подключение к GitHub

```bash
# Замените YOUR_USERNAME на ваш GitHub username
git remote add origin https://github.com/YOUR_USERNAME/RPA.git

# Или используйте SSH
git remote add origin git@github.com:YOUR_USERNAME/RPA.git
```

### 4. Push в GitHub

```bash
# Первый push
git branch -M main
git push -u origin main
```

---

## 📝 Что нужно обновить перед публикацией

### 1. README.md

Обновите следующие секции:

```markdown
## 📝 Лицензия
MIT License (или другая)

## 📞 Контакты
- Email: your.email@example.com
- GitHub: @your-username

## 🤝 Контрибьюция
См. [CONTRIBUTING.md](CONTRIBUTING.md)
```

### 2. Добавьте LICENSE

```bash
# Для MIT License
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
```

### 3. Обновите ссылки в документации

Замените `YOUR_USERNAME` на ваш GitHub username в файлах:
- `README.md`
- `CONTRIBUTING.md`
- `DEPLOYMENT.md`

```bash
# Автоматическая замена (замените YOUR_GITHUB_USERNAME)
find . -name "*.md" -type f -exec sed -i 's/YOUR_USERNAME/YOUR_GITHUB_USERNAME/g' {} +
```

---

## 🎨 GitHub настройки

### 1. Добавьте описание репозитория

```
AI-powered banking analysis suite with OperCode prediction and NER
```

### 2. Добавьте topics (теги)

```
ai, machine-learning, nlp, ner, banking, fintech, gemma, qwen, transformers, fastapi
```

### 3. Настройте GitHub Pages (опционально)

Для документации:
1. Settings → Pages
2. Source: Deploy from a branch
3. Branch: main, folder: /docs

---

## 📊 Что включено в .gitignore

Следующие файлы **НЕ** будут загружены на GitHub:

- ✅ Модели и checkpoints (слишком большие)
- ✅ Данные (CSV, Excel файлы)
- ✅ Логи и временные файлы
- ✅ Python cache и виртуальные окружения
- ✅ IDE конфигурации

### Что БУДЕТ загружено:

- ✅ Исходный код (Python скрипты)
- ✅ Документация (MD файлы)
- ✅ Конфигурационные файлы
- ✅ Скрипты запуска
- ✅ requirements.txt
- ✅ Словарь кодов (purpose_codes.txt)

---

## 📦 Структура для GitHub

```
RPA/
├── README.md                 ← Главная документация
├── LICENSE                   ← Лицензия
├── requirements.txt          ← Зависимости
├── .gitignore               ← Игнорируемые файлы
├── CONTRIBUTING.md          ← Гайд для контрибьюторов
├── DEPLOYMENT.md            ← Гайд по деплою
│
├── api/                     ← API интерфейсы
│   ├── README.md
│   ├── START_HERE.md
│   └── *.py
│
├── gemma_finetuning/        ← Fine-tuning
│   ├── README.md
│   ├── scripts/
│   └── data/ (структура, без больших файлов)
│
└── NER/                     ← NER модуль
    ├── README.md
    └── *.py
```

---

## 🔒 Безопасность

### Перед публикацией проверьте:

- [ ] Нет API ключей в коде
- [ ] Нет паролей в конфигах
- [ ] Нет персональных данных
- [ ] Нет приватной информации в логах
- [ ] .gitignore настроен правильно

### Проверка:

```bash
# Поиск потенциальных секретов
grep -r "password" --include="*.py" --include="*.md"
grep -r "api_key" --include="*.py" --include="*.md"
grep -r "secret" --include="*.py" --include="*.md"
```

---

## 📢 После публикации

### 1. Добавьте badges в README.md

```markdown
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![GPU](https://img.shields.io/badge/GPU-NVIDIA-76B900.svg)
```

### 2. Создайте Release

```bash
# Создайте тег
git tag -a v1.0.0 -m "Initial release"
git push origin v1.0.0
```

На GitHub:
1. Releases → Create a new release
2. Choose tag: v1.0.0
3. Release title: Banking Analysis Suite v1.0.0
4. Описание релиза

### 3. Настройте GitHub Actions (опционально)

Для автоматического тестирования и CI/CD.

---

## ✅ Checklist перед публикацией

- [ ] README.md заполнен и актуален
- [ ] LICENSE добавлен
- [ ] .gitignore настроен
- [ ] Нет секретов в коде
- [ ] Документация полная
- [ ] requirements.txt актуален
- [ ] Ссылки обновлены (YOUR_USERNAME → ваш username)
- [ ] Проект протестирован локально
- [ ] Git репозиторий инициализирован
- [ ] Первый commit сделан

---

## 🎉 Готово!

После выполнения всех шагов ваш проект будет опубликован на GitHub и готов к использованию сообществом!

```bash
# Финальная команда
git push -u origin main
```

**Поздравляем с публикацией!** 🚀

