"""
Evaluation скрипт для fine-tuned модели
Тестирование на test set с подробными метриками
"""

import torch
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Пути
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

def load_model(model_path):
    """Загрузка fine-tuned модели"""
    print(f"\n📥 Загрузка модели из {model_path}...")
    
    # Загружаем конфигурацию
    with open(model_path / 'training_config.json', 'r') as f:
        config = json.load(f)
    
    base_model_name = config['model_name']
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Загружаем LoRA адаптеры
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()  # Объединяем для inference
    
    print("   ✅ Модель загружена")
    
    return model, tokenizer, config

def load_test_data():
    """Загрузка тестовых данных"""
    print("\n📚 Загрузка тестовых данных...")
    
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    print(f"   Test samples: {len(test_df):,}")
    
    return test_df

def predict(model, tokenizer, text, max_new_tokens=10):
    """Предсказание для одного текста"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Извлекаем только сгенерированную часть (после промпта)
    prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    prediction = generated[prompt_length:].strip()
    
    return prediction

def extract_opercode(prediction_text):
    """Извлечение OperCode из предсказания"""
    # Пытаемся найти число в начале ответа
    import re
    match = re.search(r'\d+', prediction_text)
    if match:
        return int(match.group())
    return None

def evaluate_model(model, tokenizer, test_df):
    """Полная evaluation модели"""
    print("\n🔍 Evaluation модели на test set...")
    
    predictions = []
    true_labels = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        # Создаем промпт для inference
        prompt = f"""<start_of_turn>user
Определи код операции (OperCode) для следующего банковского платежа:

Платёж: {row['PaymentComment']}

Ответь только числовым кодом операции.<end_of_turn>
<start_of_turn>model
"""
        
        # Предсказание
        pred_text = predict(model, tokenizer, prompt)
        pred_code = extract_opercode(pred_text)
        
        predictions.append(pred_code)
        true_labels.append(row['OperCode'])
    
    # Конвертируем в numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Считаем метрики только для валидных предсказаний
    valid_mask = predictions != None
    valid_predictions = predictions[valid_mask]
    valid_labels = true_labels[valid_mask]
    
    print(f"\n   Валидных предсказаний: {valid_mask.sum()} / {len(predictions)} ({valid_mask.sum()/len(predictions)*100:.1f}%)")
    
    # Метрики
    results = {
        'accuracy': accuracy_score(valid_labels, valid_predictions),
        'f1_weighted': f1_score(valid_labels, valid_predictions, average='weighted', zero_division=0),
        'f1_macro': f1_score(valid_labels, valid_predictions, average='macro', zero_division=0),
        'precision_weighted': precision_score(valid_labels, valid_predictions, average='weighted', zero_division=0),
        'recall_weighted': recall_score(valid_labels, valid_predictions, average='weighted', zero_division=0),
        'valid_predictions_ratio': valid_mask.sum() / len(predictions),
    }
    
    return results, predictions, true_labels

def print_results(results):
    """Вывод результатов"""
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ EVALUATION")
    print("=" * 80)
    
    print(f"\n✅ Основные метрики:")
    print(f"   Accuracy:           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   F1-score (weighted): {results['f1_weighted']:.4f}")
    print(f"   F1-score (macro):    {results['f1_macro']:.4f}")
    print(f"   Precision:          {results['precision_weighted']:.4f}")
    print(f"   Recall:             {results['recall_weighted']:.4f}")
    print(f"   Valid predictions:  {results['valid_predictions_ratio']:.4f} ({results['valid_predictions_ratio']*100:.2f}%)")

def save_results(results, predictions, true_labels, output_path):
    """Сохранение результатов"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем метрики
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Сохраняем предсказания
    results_df = pd.DataFrame({
        'true_label': true_labels,
        'prediction': predictions,
        'correct': predictions == true_labels
    })
    results_df.to_csv(output_path / 'predictions.csv', index=False)
    
    print(f"\n💾 Результаты сохранены в: {output_path}")

def main():
    print("=" * 80)
    print("🎯 EVALUATION FINE-TUNED МОДЕЛИ")
    print("=" * 80)
    
    # Находим последнюю обученную модель
    model_dirs = sorted(MODEL_DIR.glob("gemma_qlora_*"))
    if not model_dirs:
        print("❌ Не найдено обученных моделей!")
        print(f"   Проверьте директорию: {MODEL_DIR}")
        return
    
    latest_model = model_dirs[-1]
    print(f"\n📁 Используется модель: {latest_model.name}")
    
    # 1. Загрузка модели
    model, tokenizer, config = load_model(latest_model)
    
    # 2. Загрузка тестовых данных
    test_df = load_test_data()
    
    # 3. Evaluation (на маленькой выборке для теста, уберите [:100] для полной оценки)
    print("\n⚠️  Запускается evaluation на первых 100 примерах (для быстрого теста)")
    print("   Для полной оценки уберите срез [:100] в коде")
    results, predictions, true_labels = evaluate_model(model, tokenizer, test_df[:100])
    
    # 4. Вывод результатов
    print_results(results)
    
    # 5. Сохранение
    output_path = OUTPUT_DIR / f"evaluation_{latest_model.name}"
    save_results(results, predictions, true_labels, output_path)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION ЗАВЕРШЕН!")
    print("=" * 80)

if __name__ == "__main__":
    main()

