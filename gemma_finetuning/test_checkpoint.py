"""
Быстрый тест checkpoint на примерах
"""
import torch
import json
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

# Настройки
CHECKPOINT_PATH = Path("outputs/gemma_qlora_20251104_181124/checkpoint-3000")
BASE_MODEL = "google/gemma-2-9b-it"
DATA_DIR = Path("data")
NUM_SAMPLES = 100  # Тестируем на 100 примерах для быстроты

print("=" * 80)
print("🧪 ТЕСТИРОВАНИЕ CHECKPOINT-3000")
print("=" * 80)

# 1. Загрузка модели
print("\n📥 Загрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("   Загрузка базовой модели...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("   Загрузка LoRA адаптеров...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
model.eval()

print("   ✅ Модель загружена")

# 2. Загрузка test данных
print("\n📚 Загрузка test данных...")
test_df = pd.read_csv(DATA_DIR / 'test.csv')
print(f"   Test примеров: {len(test_df)}")

# Берем случайную выборку
test_sample = test_df.sample(n=min(NUM_SAMPLES, len(test_df)), random_state=42)

# 3. Функция для извлечения предсказанного кода
def extract_code(text):
    """Извлекает код операции из ответа модели"""
    # Ищем числа в ответе
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[0])
    return None

# 4. Тестирование
print(f"\n🧪 Тестирование на {len(test_sample)} примерах...")
print("   (это займет ~5-10 минут)")

predictions = []
true_labels = []

for idx, row in test_sample.iterrows():
    # Создаем промпт (без ответа)
    prompt = f"""<start_of_turn>user
Определи код операции (OperCode) для следующего банковского платежа:

Платёж: {row['PaymentComment']}

Ответь только числовым кодом операции.<end_of_turn>
<start_of_turn>model
"""
    
    # Токенизация
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
        )
    
    # Декодирование
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<start_of_turn>model")[-1].strip()
    
    # Извлечение кода
    pred_code = extract_code(response)
    true_code = int(row['OperCode'])
    
    predictions.append(pred_code)
    true_labels.append(true_code)
    
    # Показываем первые 5 примеров
    if len(predictions) <= 5:
        print(f"\n   Пример {len(predictions)}:")
        print(f"   Истинный код: {true_code}")
        print(f"   Предсказано:  {pred_code}")
        print(f"   Ответ модели: {response[:100]}...")
        print(f"   {'✅ Верно' if pred_code == true_code else '❌ Неверно'}")

# 5. Вычисление метрик
print("\n" + "=" * 80)
print("📊 РЕЗУЛЬТАТЫ")
print("=" * 80)

# Фильтруем None (если модель не смогла сгенерировать код)
valid_indices = [i for i, p in enumerate(predictions) if p is not None]
valid_predictions = [predictions[i] for i in valid_indices]
valid_true_labels = [true_labels[i] for i in valid_indices]

if len(valid_predictions) > 0:
    from sklearn.metrics import accuracy_score, f1_score
    
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    f1_macro = f1_score(valid_true_labels, valid_predictions, average='macro', zero_division=0)
    f1_micro = f1_score(valid_true_labels, valid_predictions, average='micro', zero_division=0)
    
    print(f"\n📈 Метрики на {len(valid_predictions)} примерах:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1 Macro:  {f1_macro:.4f}")
    print(f"   F1 Micro:  {f1_micro:.4f}")
    
    # Анализ по частоте классов
    print(f"\n🔍 Детальный анализ:")
    
    # Загружаем веса классов
    with open('outputs/class_weights.json', 'r') as f:
        class_info = json.load(f)
    
    class_weights = class_info['class_weights']
    
    # Топ-5 и Bottom-5 классов в выборке
    from collections import Counter
    true_counts = Counter(valid_true_labels)
    
    print(f"\n   Топ-5 классов в тесте:")
    for code, count in true_counts.most_common(5):
        matches = sum(1 for p, t in zip(valid_predictions, valid_true_labels) if t == code and p == code)
        acc = matches / count if count > 0 else 0
        weight = class_weights.get(str(code), 1.0)
        print(f"      Класс {code}: {matches}/{count} верно ({acc*100:.1f}%), вес={weight:.2f}")
    
    # Процент невалидных ответов
    invalid_count = len(predictions) - len(valid_predictions)
    if invalid_count > 0:
        print(f"\n   ⚠️  Невалидных ответов: {invalid_count}/{len(predictions)} ({invalid_count/len(predictions)*100:.1f}%)")
else:
    print("\n❌ Модель не смогла сгенерировать валидные коды")

print("\n" + "=" * 80)
print("✅ ТЕСТ ЗАВЕРШЕН")
print("=" * 80)
print("\n💡 Это промежуточный результат на 18% обучения (checkpoint-3000)")
print("   Финальная модель будет намного лучше после 3 полных эпох!")

