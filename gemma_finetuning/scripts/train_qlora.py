"""
QLoRA Fine-tuning для Gemma 3:12b
Оптимизировано для 16GB VRAM
"""

import os
# Оптимизация памяти CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Отключаем предупреждения tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import json
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
import re

# ============================================================================
# КОНФИГУРАЦИЯ ДЛЯ 16GB VRAM
# ============================================================================

# Модель
MODEL_NAME = "google/gemma-2-9b-it"  # Используем Gemma 2 9B (более стабильная версия)
# Если хотите Gemma 3:12b через Ollama, нужно будет адаптировать код

# QLoRA параметры (оптимизированы для 16GB)
LORA_R = 16                    # Ранг LoRA матриц (16-32 оптимально)
LORA_ALPHA = 32                # Alpha = 2 * r (стандартная практика)
LORA_DROPOUT = 0.05            # Небольшой dropout для регуляризации
TARGET_MODULES = [             # Модули для применения LoRA
    "q_proj", "k_proj", "v_proj", 
    "o_proj", "gate_proj", "up_proj", "down_proj"
]

# Квантизация (обязательно для 16GB!)
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"

# Обучение
BATCH_SIZE = 1                 # Уменьшаем до 1 для экономии памяти
GRADIENT_ACCUMULATION = 16     # Увеличиваем accumulation (эффективный batch = 1 * 16 = 16)
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 384           # Уменьшаем с 512 до 384
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.001

# Оптимизация памяти
GRADIENT_CHECKPOINTING = True
OPTIM = "paged_adamw_8bit"    # Оптимизатор с пагинацией

# Балансировка классов
USE_CLASS_WEIGHTS = True        # Использовать веса классов
USE_WEIGHTED_SAMPLER = True     # Использовать взвешенную выборку
FOCAL_LOSS_GAMMA = 2.0          # Параметр focal loss (0 = обычная CE loss)

# Пути
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# ============================================================================
# ФУНКЦИИ
# ============================================================================

def print_gpu_memory():
    """Вывод информации о памяти GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   💾 GPU Memory: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total")

def extract_oper_code(text):
    """Извлечение OperCode из текста промпта"""
    # Ищем код после <start_of_turn>model
    match = re.search(r'<start_of_turn>model\s*(\d+)', text)
    if match:
        return int(match.group(1))
    # Запасной вариант - ищем любое число в конце
    match = re.search(r'(\d+)\s*<end_of_turn>', text)
    if match:
        return int(match.group(1))
    return None

def compute_class_weights_and_sampler(dataset):
    """
    Вычисление весов классов и создание WeightedRandomSampler
    """
    print("\n⚖️  Вычисление весов классов...")
    
    # Извлекаем все коды операций из датасета
    oper_codes = []
    for item in dataset:
        code = extract_oper_code(item['text'])
        if code is not None:
            oper_codes.append(code)
    
    oper_codes = np.array(oper_codes)
    unique_classes = np.unique(oper_codes)
    
    print(f"   Найдено классов: {len(unique_classes)}")
    print(f"   Примеров: {len(oper_codes)}")
    
    # Вычисляем веса классов (обратно пропорционально частоте)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=oper_codes
    )
    
    # Создаем словарь класс -> вес
    class_weights_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights)}
    
    # Статистика
    min_weight = min(class_weights_dict.values())
    max_weight = max(class_weights_dict.values())
    print(f"   Веса классов: от {min_weight:.4f} до {max_weight:.4f}")
    
    # Создаем веса для каждого примера (для WeightedRandomSampler)
    sample_weights = [class_weights_dict[code] for code in oper_codes]
    
    # Создаем sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"   ✅ WeightedRandomSampler создан")
    
    return class_weights_dict, sampler, unique_classes

def setup_quantization_config():
    """Настройка конфигурации квантизации"""
    if USE_4BIT:
        compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,  # Двойная квантизация для экономии
        )
        return bnb_config
    return None

def load_model_and_tokenizer():
    """Загрузка модели и токенизатора"""
    print("\n📥 Загрузка модели и токенизатора...")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"   ✅ Токенизатор загружен: {MODEL_NAME}")
    
    # Quantization config
    bnb_config = setup_quantization_config()
    
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Используем стандартную реализацию (flash_attn не установлен)
    )
    
    print(f"   ✅ Модель загружена: {MODEL_NAME}")
    print_gpu_memory()
    
    # Подготовка для k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)
    
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("   ✅ Gradient checkpointing включен")
    
    return model, tokenizer

def setup_lora(model):
    """Настройка LoRA адаптеров"""
    print("\n🔧 Настройка LoRA адаптеров...")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Статистика trainable параметров
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params
    
    print(f"   📊 Параметры:")
    print(f"      Всего: {all_params:,}")
    print(f"      Обучаемых: {trainable_params:,} ({trainable_percent:.2f}%)")
    print(f"   ✅ LoRA настроен (r={LORA_R}, alpha={LORA_ALPHA})")
    
    return model

def load_and_prepare_datasets(tokenizer):
    """Загрузка и подготовка датасетов"""
    print("\n📚 Загрузка датасетов...")
    
    # Загрузка
    dataset = load_dataset(
        'json',
        data_files={
            'train': str(DATA_DIR / 'train.jsonl'),
            'validation': str(DATA_DIR / 'val.jsonl'),
            'test': str(DATA_DIR / 'test.jsonl')
        }
    )
    
    print(f"   Train: {len(dataset['train']):,}")
    print(f"   Val:   {len(dataset['validation']):,}")
    print(f"   Test:  {len(dataset['test']):,}")
    
    # Токенизация
    def tokenize_function(examples):
        # Токенизируем тексты (промпты)
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Labels для языковой модели (копия input_ids)
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        return tokenized
    
    print("\n🔄 Токенизация датасетов...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )
    
    print("   ✅ Токенизация завершена")
    
    return tokenized_datasets

class WeightedLossTrainer(Trainer):
    """
    Custom Trainer с поддержкой weighted loss и focal loss
    """
    def __init__(self, class_weights_dict=None, focal_gamma=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_dict = class_weights_dict
        self.focal_gamma = focal_gamma
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Переопределяем compute_loss для добавления class weights и focal loss
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift для causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Маска для игнорирования padding
        mask = shift_labels != -100
        
        if self.class_weights_dict is not None and USE_CLASS_WEIGHTS:
            # Применяем class weights
            # Для токенов мы не можем применить веса напрямую,
            # поэтому используем стандартную CE loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits, shift_labels)
            
            # Применяем focal loss, если нужно
            if self.focal_gamma > 0:
                ce_loss = loss
                pt = torch.exp(-ce_loss)
                focal_weight = (1 - pt) ** self.focal_gamma
                loss = focal_weight * ce_loss
            
            # Усредняем только по не-padding токенам
            loss = loss[mask].mean()
        else:
            # Стандартная loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """Вычисление метрик для evaluation"""
    logits, labels = eval_pred
    
    # Получаем предсказания (argmax по последнему измерению)
    predictions = np.argmax(logits, axis=-1)
    
    # Фильтруем padding tokens (-100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Базовые метрики
    accuracy = accuracy_score(labels, predictions)
    
    # Macro/Micro F1 (игнорируем warning для редких классов)
    try:
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    except:
        f1_macro = 0.0
        f1_micro = 0.0
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
    }

def setup_training_arguments():
    """Настройка аргументов обучения"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"gemma_qlora_{timestamp}"
    
    training_args = TrainingArguments(
        # Пути
        output_dir=str(OUTPUT_DIR / run_name),
        logging_dir=str(LOGS_DIR / run_name),
        
        # Обучение
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=4,  # Увеличиваем до 4 (evaluation в 4 раза быстрее!)
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        
        # Оптимизация
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        optim=OPTIM,
        
        # Сохранение
        save_strategy="steps",
        save_steps=1000,  # Совпадает с eval_steps (требование load_best_model_at_end)
        save_total_limit=3,
        
        # Evaluation
        eval_strategy="steps",  # Было evaluation_strategy в старых версиях
        eval_steps=1000,  # Увеличиваем с 500 до 1000 (меньше OOM)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Меняем на eval_loss (не требует вычисления метрик)
        greater_is_better=False,
        prediction_loss_only=True,  # Сохраняем только loss, не все logits
        
        # Логирование
        logging_steps=50,
        logging_first_step=True,
        
        # Оптимизация памяти
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        fp16=False,
        bf16=True,  # BFloat16 для лучшей стабильности
        
        # Прочее
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=2,  # Уменьшаем с 4 до 2 для экономии памяти
        dataloader_pin_memory=False,  # Отключаем pin_memory для экономии
        max_grad_norm=1.0,  # Gradient clipping
    )
    
    return training_args, run_name

def train():
    """Основная функция обучения"""
    print("=" * 80)
    print("🚀 QLORA FINE-TUNING GEMMA 3:12B")
    print("=" * 80)
    print(f"\n⚙️  Конфигурация:")
    print(f"   Модель: {MODEL_NAME}")
    print(f"   LoRA r: {LORA_R}, alpha: {LORA_ALPHA}")
    print(f"   Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Max seq length: {MAX_SEQ_LENGTH}")
    print(f"   4-bit quantization: {USE_4BIT}")
    print(f"\n⚖️  Балансировка классов:")
    print(f"   Class weights: {USE_CLASS_WEIGHTS}")
    print(f"   Weighted sampler: {USE_WEIGHTED_SAMPLER}")
    print(f"   Focal loss gamma: {FOCAL_LOSS_GAMMA}")
    
    # 1. Загрузка модели
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. Настройка LoRA
    model = setup_lora(model)
    
    # 3. Загрузка данных
    tokenized_datasets = load_and_prepare_datasets(tokenizer)
    
    # 3.5. Вычисление весов классов и sampler
    class_weights_dict = None
    train_sampler = None
    unique_classes = None
    
    if USE_CLASS_WEIGHTS or USE_WEIGHTED_SAMPLER:
        # Загружаем исходные данные для вычисления весов
        raw_dataset = load_dataset(
            'json',
            data_files={'train': str(DATA_DIR / 'train.jsonl')}
        )
        
        class_weights_dict, train_sampler, unique_classes = compute_class_weights_and_sampler(
            raw_dataset['train']
        )
        
        # Сохраняем веса классов
        weights_save_path = OUTPUT_DIR / 'class_weights.json'
        weights_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(weights_save_path, 'w') as f:
            json.dump({
                'class_weights': class_weights_dict,
                'num_classes': len(unique_classes),
                'classes': unique_classes.tolist()
            }, f, indent=2)
        print(f"   💾 Веса классов сохранены: {weights_save_path}")
    
    # 4. Настройка обучения
    training_args, run_name = setup_training_arguments()
    
    # 5. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, не masked LM
    )
    
    # 6. Trainer
    print("\n🏋️  Инициализация Trainer...")
    
    # Используем WeightedLossTrainer если нужны веса классов
    trainer_class = WeightedLossTrainer if USE_CLASS_WEIGHTS else Trainer
    
    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_datasets['train'],
        'eval_dataset': tokenized_datasets['validation'],
        'data_collator': data_collator,
        # compute_metrics убран - используем prediction_loss_only=True для экономии памяти
    }
    
    # Добавляем параметры для WeightedLossTrainer
    if USE_CLASS_WEIGHTS:
        trainer_kwargs['class_weights_dict'] = class_weights_dict
        trainer_kwargs['focal_gamma'] = FOCAL_LOSS_GAMMA
    
    trainer = trainer_class(**trainer_kwargs)
    
    # Note: WeightedRandomSampler нужно интегрировать через DataLoader,
    # что требует более глубокой кастомизации. Пока используем только weighted loss.
    
    print("   ✅ Trainer готov")
    if USE_CLASS_WEIGHTS:
        print(f"   ⚖️  Используются веса классов")
    if FOCAL_LOSS_GAMMA > 0:
        print(f"   🎯 Focal Loss с gamma={FOCAL_LOSS_GAMMA}")
    print(f"   📊 Evaluation каждые 1000 шагов (только loss для экономии памяти)")
    
    # 7. Обучение
    print("\n" + "=" * 80)
    print("🎓 НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 80)
    
    # Очищаем кэш GPU перед началом
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("   🧹 GPU cache очищен")
    
    print_gpu_memory()
    
    # Проверяем, есть ли checkpoint для продолжения (ищем во всех папках)
    resume_checkpoint = None
    all_checkpoints = []
    for run_dir in OUTPUT_DIR.glob("gemma_qlora_*"):
        if run_dir.is_dir():
            for ckpt in run_dir.glob("checkpoint-*"):
                if ckpt.is_dir():
                    step = int(ckpt.name.split("-")[1])
                    all_checkpoints.append((step, ckpt))
    
    if all_checkpoints:
        # Берем самый последний checkpoint
        all_checkpoints.sort(key=lambda x: x[0])
        latest_step, latest_ckpt = all_checkpoints[-1]
        resume_checkpoint = str(latest_ckpt)
        print(f"\n🔄 Найден checkpoint для продолжения: {latest_ckpt.name}")
        print(f"   📂 Путь: {latest_ckpt}")
        print(f"   ⏭️  Обучение продолжится с шага {latest_step}")
    
    try:
        if resume_checkpoint:
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            trainer.train()
        
        print("\n" + "=" * 80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        raise
    
    # 8. Сохранение модели
    print("\n💾 Сохранение модели...")
    
    model_save_path = MODEL_DIR / run_name
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем только LoRA адаптеры (легкий вес!)
    trainer.model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Сохраняем конфигурацию
    config = {
        'model_name': MODEL_NAME,
        'lora_r': LORA_R,
        'lora_alpha': LORA_ALPHA,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_epochs': NUM_EPOCHS,
        'training_completed': datetime.now().isoformat(),
    }
    
    with open(model_save_path / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   ✅ Модель сохранена в: {model_save_path}")
    print(f"   📦 Размер LoRA адаптеров: ~200-500 MB")
    
    # 9. Финальная evaluation
    print("\n📊 Финальная оценка на validation set...")
    eval_results = trainer.evaluate()
    
    print("\n📈 Основные метрики:")
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")
    
    # Сохраняем результаты
    results_path = model_save_path / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n   💾 Результаты сохранены: {results_path}")
    
    # 10. Детальная оценка по классам (если доступны веса)
    if class_weights_dict is not None:
        print("\n📊 Анализ распределения классов:")
        
        # Топ-5 по количеству примеров
        print("\n   🔝 Топ-5 самых частых классов:")
        class_counts = {}
        for code in unique_classes:
            # Подсчитываем из исходных данных
            class_counts[int(code)] = 1.0 / class_weights_dict.get(int(code), 1.0)
        
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for code, _ in top_classes:
            weight = class_weights_dict.get(code, 1.0)
            print(f"      Класс {code}: вес={weight:.4f}")
        
        # Топ-5 самых редких
        print("\n   🔻 Топ-5 самых редких классов:")
        bottom_classes = sorted(class_counts.items(), key=lambda x: x[1])[:5]
        for code, _ in bottom_classes:
            weight = class_weights_dict.get(code, 1.0)
            print(f"      Класс {code}: вес={weight:.4f}")
    
    print("\n" + "=" * 80)
    print(f"🎉 ВСЁ ГОТОВО! Модель: {model_save_path}")
    print("=" * 80)
    print(f"\n🔍 Следующие шаги:")
    print(f"   1. Посмотрите логи в: {LOGS_DIR / run_name}")
    print(f"   2. Запустите evaluation: python scripts/evaluate.py")
    print(f"   3. Тестируйте inference: python scripts/inference.py")
    print(f"\n💡 Рекомендации:")
    print(f"   • Во время обучения отслеживался только eval_loss (экономия памяти)")
    print(f"   • Запустите evaluate.py для полных метрик (F1, Accuracy, per-class)")
    print(f"   • evaluate.py покажет детальную статистику по всем 72 классам")
    
    return model_save_path

if __name__ == "__main__":
    train()

