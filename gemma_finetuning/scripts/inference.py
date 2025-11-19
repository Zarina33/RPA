"""
Inference скрипт для использования обученной модели
Предсказание OperCode для новых платежей
"""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

# Пути
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"

class OperCodeClassifier:
    """Класс для классификации платежей"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.config = None
        self.load_model()
    
    def load_model(self):
        """Загрузка модели"""
        print(f"📥 Загрузка модели из {self.model_path.name}...")
        
        # Конфигурация
        with open(self.model_path / 'training_config.json', 'r') as f:
            self.config = json.load(f)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()
        
        print("   ✅ Модель готова к работе")
    
    def predict(self, payment_comment, return_confidence=False):
        """
        Предсказание OperCode для платежа
        
        Args:
            payment_comment (str): Текст платежа
            return_confidence (bool): Возвращать ли уверенность модели
            
        Returns:
            int or tuple: OperCode или (OperCode, confidence)
        """
        # Создаем промпт
        prompt = f"""<start_of_turn>user
Определи код операции (OperCode) для следующего банковского платежа:

Платёж: {payment_comment}

Ответь только числовым кодом операции.<end_of_turn>
<start_of_turn>model
"""
        
        # Токенизация
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Декодирование
        generated = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Извлекаем ответ модели
        prompt_length = len(self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
        prediction_text = generated[prompt_length:].strip()
        
        # Извлекаем число
        opercode = self._extract_opercode(prediction_text)
        
        if return_confidence:
            # Вычисляем уверенность (упрощенно)
            confidence = self._calculate_confidence(outputs.scores)
            return opercode, confidence
        
        return opercode
    
    def _extract_opercode(self, text):
        """Извлечение OperCode из текста"""
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        return None
    
    def _calculate_confidence(self, scores):
        """Вычисление уверенности модели"""
        if not scores:
            return 0.0
        
        # Берем softmax от первого токена ответа
        first_token_scores = torch.softmax(scores[0][0], dim=-1)
        confidence = first_token_scores.max().item()
        
        return confidence
    
    def predict_batch(self, payment_comments):
        """
        Предсказание для batch платежей
        
        Args:
            payment_comments (list): Список текстов платежей
            
        Returns:
            list: Список OperCode
        """
        results = []
        for comment in payment_comments:
            opercode = self.predict(comment)
            results.append(opercode)
        return results

def find_latest_model():
    """Поиск последней обученной модели"""
    model_dirs = sorted(MODEL_DIR.glob("gemma_qlora_*"))
    if not model_dirs:
        raise FileNotFoundError(f"Не найдено обученных моделей в {MODEL_DIR}")
    return model_dirs[-1]

def interactive_mode(classifier):
    """Интерактивный режим"""
    print("\n" + "=" * 80)
    print("🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 80)
    print("Введите текст платежа для классификации (или 'exit' для выхода)")
    print("-" * 80)
    
    while True:
        print("\n💬 Введите платёж:")
        payment = input("> ").strip()
        
        if payment.lower() in ['exit', 'quit', 'q']:
            print("👋 До свидания!")
            break
        
        if not payment:
            continue
        
        # Предсказание
        opercode, confidence = classifier.predict(payment, return_confidence=True)
        
        print(f"\n📊 Результат:")
        print(f"   OperCode: {opercode}")
        print(f"   Уверенность: {confidence:.2%}")

def demo_mode(classifier):
    """Демо режим с примерами"""
    print("\n" + "=" * 80)
    print("🎬 ДЕМО РЕЖИМ")
    print("=" * 80)
    
    examples = [
        "'TRANSFER OF FUNDS TO OWN ACCOUNT",
        "'PURCHASE OF GOODS",
        "PAYMENT FOR SALES AGENT SERVICES BY CONT B N DD 21 10 2024",
        "OPLATA ZA TRANSPORTNYE USLUGI PO ScET U 000390",
        "POLUcATELX 'Feldman inna' ScET PODAROK NEREZIDENTU",
    ]
    
    print("\nПримеры классификации:\n")
    
    for i, example in enumerate(examples, 1):
        opercode, confidence = classifier.predict(example, return_confidence=True)
        print(f"{i}. Платёж: {example[:70]}...")
        print(f"   → OperCode: {opercode} (уверенность: {confidence:.2%})\n")

def main():
    print("=" * 80)
    print("🚀 OPERCODE CLASSIFIER - INFERENCE")
    print("=" * 80)
    
    # Поиск модели
    try:
        model_path = find_latest_model()
        print(f"\n📁 Используется модель: {model_path.name}")
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        return
    
    # Загрузка классификатора
    classifier = OperCodeClassifier(model_path)
    
    # Выбор режима
    print("\n📋 Выберите режим:")
    print("   1. Демо (примеры)")
    print("   2. Интерактивный (ввод с клавиатуры)")
    print("   3. Оба")
    
    choice = input("\nВыбор (1/2/3): ").strip()
    
    if choice == '1':
        demo_mode(classifier)
    elif choice == '2':
        interactive_mode(classifier)
    elif choice == '3':
        demo_mode(classifier)
        interactive_mode(classifier)
    else:
        print("❌ Неверный выбор")

if __name__ == "__main__":
    main()

