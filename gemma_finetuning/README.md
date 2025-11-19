# 🚀 QLoRA Fine-tuning Gemma 3:12b для Классификации Банковских Платежей

Fine-tuning Gemma 3:12b с использованием QLoRA для задачи классификации `PaymentComment → OperCode`.

## 📊 Задача

**Вход:** Текст банковского платежа (PaymentComment)  
**Выход:** Код операции (OperCode) - 86 классов

**Данные:** 388,706 транзакций из SWIFT система

## 🎯 Особенности

- ✅ **QLoRA** - эффективное fine-tuning для 16GB VRAM
- ✅ **4-bit квантизация** - экономия памяти
- ✅ **Gradient checkpointing** - дополнительная оптимизация
- ✅ **Stratified split** - сбалансированное разделение данных
- ✅ **Мультиязычность** - поддержка русского и английского

## 💻 Требования

### Железо
- **GPU:** 16GB VRAM (RTX 3060 Ti, 4060 Ti, 3090, 4090)
- **RAM:** 32GB рекомендуется
- **Диск:** ~50GB свободного места

### Софт
- Python 3.10+
- CUDA 11.8+ или 12.1+
- Linux (рекомендуется) или WSL2

## 📦 Установка

### 1. Клонирование и переход в директорию
```bash
cd /home/zarina/Work/RPA/gemma_finetuning
```

### 2. Создание виртуального окружения
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### 3. Установка зависимостей

```bash
# Сначала PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Затем остальные зависимости
pip install -r requirements.txt
```

**Проверка GPU:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## 🏃 Быстрый старт

### Шаг 1: Подготовка данных

```bash
cd scripts
python prepare_data.py
```

**Что делает:**
- Загружает `final_dataset.csv`
- Создает train/val/test split (70/15/15)
- Генерирует промпты для Gemma
- Сохраняет в `data/` в формате CSV и JSONL

**Выход:**
```
data/
├── train.csv & train.jsonl  (~272k примеров)
├── val.csv & val.jsonl      (~58k примеров)
├── test.csv & test.jsonl    (~58k примеров)
└── metadata.json
```

### Шаг 2: Fine-tuning

```bash
python train_qlora.py
```

**Параметры (настраиваются в скрипте):**
- Batch size: 2 x 8 accumulation = 16 эффективный
- Learning rate: 2e-4
- Epochs: 3
- LoRA r=16, alpha=32

**Время:** ~1-2 дня на RTX 3060 Ti

**Мониторинг в реальном времени:**
```bash
# В другом терминале
tensorboard --logdir=logs/
```
Открыть: http://localhost:6006

**Что сохраняется:**
```
models/gemma_qlora_YYYYMMDD_HHMMSS/
├── adapter_config.json
├── adapter_model.safetensors  (~200-500MB)
├── training_config.json
└── tokenizer/
```

### Шаг 3: Evaluation

```bash
python evaluate.py
```

**Метрики:**
- Accuracy
- F1-score (weighted & macro)
- Precision & Recall
- Classification report

**Выход:**
```
outputs/evaluation_gemma_qlora_YYYYMMDD_HHMMSS/
├── evaluation_results.json
└── predictions.csv
```

### Шаг 4: Inference

```bash
python inference.py
```

**Режимы:**
1. **Демо** - примеры классификации
2. **Интерактивный** - ввод с клавиатуры
3. **Оба**

**Пример использования в коде:**
```python
from inference import OperCodeClassifier

# Загрузка модели
classifier = OperCodeClassifier("models/gemma_qlora_YYYYMMDD_HHMMSS")

# Предсказание
payment = "TRANSFER OF FUNDS TO OWN ACCOUNT"
opercode = classifier.predict(payment)
print(f"OperCode: {opercode}")

# С уверенностью
opercode, confidence = classifier.predict(payment, return_confidence=True)
print(f"OperCode: {opercode} (confidence: {confidence:.2%})")

# Batch
payments = ["...", "...", "..."]
opcodes = classifier.predict_batch(payments)
```

## 🔧 Настройка под вашу GPU

### Если 12GB VRAM
В `train_qlora.py` измените:
```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
MAX_SEQ_LENGTH = 384
```

### Если 24GB+ VRAM
```python
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LORA_R = 32
LORA_ALPHA = 64
```

### Если нет GPU - Google Colab
1. Colab Pro ($10/месяц) - RTX A100 40GB
2. Загрузите проект в Colab
3. Установите зависимости
4. Запустите обучение

## 📊 Ожидаемые результаты

| Метрика | Ожидаемое значение |
|---------|-------------------|
| Accuracy | 90-95% |
| F1-score (weighted) | 89-94% |
| F1-score (macro) | 75-85% |
| Training time | 1-2 дня (16GB GPU) |
| Inference time | ~1-2 сек/пример |

**Примечание:** Macro F1 ниже из-за сильного дисбаланса классов.

## 📁 Структура проекта

```
gemma_finetuning/
├── data/                  # Подготовленные датасеты
│   ├── train.csv/jsonl
│   ├── val.csv/jsonl
│   ├── test.csv/jsonl
│   └── metadata.json
├── models/                # Сохраненные модели (LoRA адаптеры)
│   └── gemma_qlora_*/
├── outputs/               # Результаты evaluation
│   └── evaluation_*/
├── logs/                  # TensorBoard логи
│   └── gemma_qlora_*/
├── scripts/               # Скрипты
│   ├── prepare_data.py    # Подготовка данных
│   ├── train_qlora.py     # Обучение
│   ├── evaluate.py        # Оценка
│   └── inference.py       # Использование модели
├── requirements.txt       # Зависимости
└── README.md             # Эта инструкция
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# В train_qlora.py:
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
MAX_SEQ_LENGTH = 256
```

### Медленное обучение
1. Проверьте, что используется GPU: `watch -n 1 nvidia-smi`
2. Установите flash-attention: `pip install flash-attn --no-build-isolation`
3. Уменьшите `MAX_SEQ_LENGTH`

### Ошибка при загрузке модели
- Убедитесь, что установлен правильный PyTorch для вашей CUDA
- Проверьте доступ к HuggingFace: `huggingface-cli login`

### Низкая точность
1. Увеличьте `NUM_EPOCHS` (3 → 5)
2. Увеличьте `LORA_R` (16 → 32)
3. Уменьшите `LEARNING_RATE` (2e-4 → 1e-4)
4. Добавьте больше данных для редких классов

## 🎓 Дополнительные материалы

- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [Gemma documentation](https://ai.google.dev/gemma)
- [PEFT documentation](https://huggingface.co/docs/peft)
- [Transformers documentation](https://huggingface.co/docs/transformers)

## 📧 Контакты

Проект: RPA SWIFT Transaction Classification  
Дата: 2025-11-04

## 📝 Лицензия

Используйте в соответствии с лицензией Gemma и вашими корпоративными правилами.

---

**Удачи в обучении! 🚀**

