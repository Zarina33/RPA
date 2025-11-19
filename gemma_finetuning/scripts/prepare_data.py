"""
Подготовка данных для fine-tuning Gemma 3:12b
Задача: Классификация PaymentComment → OperCode
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path
from collections import Counter

# Настройки
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
MIN_SAMPLES_PER_CLASS = 10  # Минимальное количество примеров для класса

def load_dataset(file_path):
    """Загрузка исходного датасета"""
    print(f"📥 Загрузка данных из {file_path}...")
    df = pd.read_csv(file_path)
    print(f"   Загружено: {len(df):,} записей")
    return df

def analyze_class_distribution(df, column='OperCode'):
    """Анализ распределения классов"""
    print(f"\n📊 Анализ распределения классов ({column}):")
    class_counts = df[column].value_counts()
    print(f"   Уникальных классов: {len(class_counts)}")
    print(f"   Мин/Макс примеров: {class_counts.min()} / {class_counts.max():,}")
    
    rare_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]
    print(f"   ⚠️  Классов с < {MIN_SAMPLES_PER_CLASS} примеров: {len(rare_classes)}")
    
    return class_counts

def filter_rare_classes(df, min_samples=MIN_SAMPLES_PER_CLASS):
    """Опционально: фильтрация редких классов"""
    class_counts = df['OperCode'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    
    df_filtered = df[df['OperCode'].isin(valid_classes)].copy()
    removed = len(df) - len(df_filtered)
    
    if removed > 0:
        print(f"\n🔧 Фильтрация редких классов:")
        print(f"   Удалено {removed:,} записей ({removed/len(df)*100:.2f}%)")
        print(f"   Осталось классов: {df_filtered['OperCode'].nunique()}")
    
    return df_filtered

def prepare_text(text):
    """Подготовка текста (базовая очистка)"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Убираем лишние пробелы
    text = ' '.join(text.split())
    return text

def create_prompt(payment_comment, oper_code=None, for_training=True):
    """
    Создание промпта для Gemma 3
    
    Формат для обучения:
    <start_of_turn>user
    Классифицируй платёж: [текст]
    <end_of_turn>
    <start_of_turn>model
    [код]<end_of_turn>
    """
    if for_training:
        prompt = f"""<start_of_turn>user
Определи код операции (OperCode) для следующего банковского платежа:

Платёж: {payment_comment}

Ответь только числовым кодом операции.<end_of_turn>
<start_of_turn>model
{oper_code}<end_of_turn>"""
    else:
        prompt = f"""<start_of_turn>user
Определи код операции (OperCode) для следующего банковского платежа:

Платёж: {payment_comment}

Ответь только числовым кодом операции.<end_of_turn>
<start_of_turn>model
"""
    
    return prompt

def prepare_for_training(df):
    """Подготовка данных в формат для обучения"""
    print(f"\n🔧 Подготовка данных для обучения...")
    
    # Очистка текстов
    df['PaymentComment'] = df['PaymentComment'].apply(prepare_text)
    
    # Создание промптов
    df['prompt'] = df.apply(
        lambda row: create_prompt(row['PaymentComment'], row['OperCode'], for_training=True),
        axis=1
    )
    
    # Создание меток
    df['label'] = df['OperCode'].astype(str)
    
    print(f"   ✅ Подготовлено {len(df):,} примеров")
    
    return df

def split_dataset(df):
    """Разделение на train/val/test с stratification"""
    print(f"\n✂️  Разделение датасета...")
    
    # Сначала отделяем test
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df['OperCode']
    )
    
    # Затем из train_val выделяем validation
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        random_state=RANDOM_SEED,
        stratify=train_val['OperCode']
    )
    
    print(f"   📚 Train: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
    print(f"   📖 Val:   {len(val):,} ({len(val)/len(df)*100:.1f}%)")
    print(f"   📝 Test:  {len(test):,} ({len(test)/len(df)*100:.1f}%)")
    
    return train, val, test

def save_datasets(train, val, test, output_dir):
    """Сохранение датасетов"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Сохранение датасетов в {output_dir}...")
    
    # Сохраняем в формате CSV
    train[['prompt', 'label', 'OperCode', 'PaymentComment']].to_csv(
        output_dir / 'train.csv', index=False
    )
    val[['prompt', 'label', 'OperCode', 'PaymentComment']].to_csv(
        output_dir / 'val.csv', index=False
    )
    test[['prompt', 'label', 'OperCode', 'PaymentComment']].to_csv(
        output_dir / 'test.csv', index=False
    )
    
    # Сохраняем в формате JSONL (для transformers)
    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        with open(output_dir / f'{split_name}.jsonl', 'w', encoding='utf-8') as f:
            for _, row in split_df.iterrows():
                json_obj = {
                    'text': row['prompt'],
                    'label': row['label'],
                    'oper_code': int(row['OperCode']),
                    'payment_comment': row['PaymentComment']
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Сохранено: train.csv, val.csv, test.csv")
    print(f"   ✅ Сохранено: train.jsonl, val.jsonl, test.jsonl")
    
    # Сохраняем метаинформацию
    metadata = {
        'total_samples': len(train) + len(val) + len(test),
        'train_samples': len(train),
        'val_samples': len(val),
        'test_samples': len(test),
        'num_classes': train['OperCode'].nunique(),
        'class_distribution': train['OperCode'].value_counts().to_dict(),
        'min_samples_per_class': MIN_SAMPLES_PER_CLASS,
        'random_seed': RANDOM_SEED
    }
    
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Сохранено: metadata.json")

def main():
    print("=" * 80)
    print("🚀 ПОДГОТОВКА ДАННЫХ ДЛЯ QLORA FINE-TUNING GEMMA 3:12B")
    print("=" * 80)
    
    # Пути
    input_file = Path(__file__).parent.parent.parent / 'final_dataset.csv'
    output_dir = Path(__file__).parent.parent / 'data'
    
    # 1. Загрузка
    df = load_dataset(input_file)
    
    # 2. Анализ
    analyze_class_distribution(df)
    
    # 3. Фильтрация редких классов (необходимо для стратификации)
    print("\n🔧 Фильтрация редких классов...")
    print(f"   Удаляем классы с < {MIN_SAMPLES_PER_CLASS} примеров (требуется для стратификации)")
    df_filtered = filter_rare_classes(df, min_samples=MIN_SAMPLES_PER_CLASS)
    print(f"   ✅ Осталось {df_filtered['OperCode'].nunique()} классов")
    
    # 4. Подготовка
    df_prepared = prepare_for_training(df_filtered)
    
    # 5. Разделение
    train, val, test = split_dataset(df_prepared)
    
    # 6. Сохранение
    save_datasets(train, val, test, output_dir)
    
    # 7. Итоговая статистика
    print("\n" + "=" * 80)
    print("✅ ПОДГОТОВКА ЗАВЕРШЕНА!")
    print("=" * 80)
    print(f"\n📁 Датасеты сохранены в: {output_dir}")
    print(f"\n📊 Итоговая статистика:")
    print(f"   Классов: {train['OperCode'].nunique()}")
    print(f"   Train:   {len(train):,}")
    print(f"   Val:     {len(val):,}")
    print(f"   Test:    {len(test):,}")
    print(f"\n🎯 Следующий шаг: Запустите scripts/train_qlora.py")
    print("=" * 80)

if __name__ == "__main__":
    main()

