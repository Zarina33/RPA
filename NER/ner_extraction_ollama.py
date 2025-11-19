"""
Named Entity Recognition (NER) для извлечения имени и фамилии из текста
Использует модель qwen3:14b через Ollama Python SDK
"""

import json
import re
from typing import Dict, Optional, List
import ollama


class NameExtractorOllama:
    """Класс для извлечения имени и фамилии из текста используя Ollama SDK"""
    
    def __init__(self, model_name: str = "qwen2.5:14b"):
        """
        Инициализация экстрактора имен
        
        Args:
            model_name: Название модели в Ollama
        """
        self.model_name = model_name
        self._verify_model()
    
    def _verify_model(self):
        """Проверка доступности модели"""
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]
            if not any(self.model_name in name for name in model_names):
                print(f"⚠️  Модель {self.model_name} не найдена локально.")
                print(f"   Доступные модели: {', '.join(model_names)}")
                print(f"   Для загрузки модели выполните: ollama pull {self.model_name}")
        except Exception as e:
            print(f"⚠️  Не удалось проверить доступность модели: {e}")
    
    def _create_system_prompt(self) -> str:
        """Создает системный промпт"""
        return """Ты - система извлечения именованных сущностей (NER). Твоя задача - извлечь из текста имя и фамилию человека.

Правила работы:
1. Ищи имя и фамилию человека в тексте
2. Возвращай результат ТОЛЬКО в формате JSON
3. Если имя или фамилия не найдены, используй null
4. В remaining_text убери имя и фамилию, но сохрани остальной текст
5. Не добавляй никаких дополнительных комментариев"""
    
    def _create_user_prompt(self, text: str) -> str:
        """
        Создает пользовательский промпт
        
        Args:
            text: Входной текст для анализа
            
        Returns:
            Форматированный промпт
        """
        return f"""Текст для анализа: "{text}"

Верни результат в формате JSON:
{{
    "first_name": "имя или null",
    "last_name": "фамилия или null",
    "remaining_text": "остальной текст без имени и фамилии"
}}"""
    
    def _call_model(self, text: str, temperature: float = 0.1) -> str:
        """
        Вызов модели через Ollama SDK
        
        Args:
            text: Текст для анализа
            temperature: Температура генерации
            
        Returns:
            Ответ модели
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': self._create_system_prompt()
                    },
                    {
                        'role': 'user',
                        'content': self._create_user_prompt(text)
                    }
                ],
                options={
                    'temperature': temperature,
                    'num_predict': 500
                }
            )
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Ошибка при вызове модели: {str(e)}")
    
    def _parse_response(self, response: str) -> Dict[str, Optional[str]]:
        """
        Парсинг ответа модели
        
        Args:
            response: Ответ модели
            
        Returns:
            Словарь с извлеченными данными
        """
        # Очистка ответа от markdown
        response = response.replace('```json', '').replace('```', '').strip()
        
        # Попытка извлечь JSON из ответа
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Обработка null значений
                first_name = data.get("first_name")
                last_name = data.get("last_name")
                
                if first_name == "null" or first_name == "None":
                    first_name = None
                if last_name == "null" or last_name == "None":
                    last_name = None
                
                return {
                    "first_name": first_name,
                    "last_name": last_name,
                    "remaining_text": data.get("remaining_text", "").strip()
                }
            except json.JSONDecodeError as e:
                print(f"⚠️  Ошибка парсинга JSON: {e}")
                print(f"   Ответ модели: {response}")
        
        # Если не удалось распарсить JSON, возвращаем пустой результат
        return {
            "first_name": None,
            "last_name": None,
            "remaining_text": response
        }
    
    def extract(self, text: str) -> Dict[str, Optional[str]]:
        """
        Извлечение имени и фамилии из текста
        
        Args:
            text: Входной текст
            
        Returns:
            Словарь с полями:
                - first_name: имя
                - last_name: фамилия
                - remaining_text: остальной текст
        """
        if not text or not text.strip():
            return {
                "first_name": None,
                "last_name": None,
                "remaining_text": ""
            }
        
        # Вызов модели
        response = self._call_model(text)
        
        # Парсинг ответа
        result = self._parse_response(response)
        
        return result
    
    def extract_batch(self, texts: List[str]) -> List[Dict[str, Optional[str]]]:
        """
        Пакетное извлечение имен и фамилий
        
        Args:
            texts: Список текстов для обработки
            
        Returns:
            Список результатов
        """
        results = []
        for i, text in enumerate(texts, 1):
            print(f"Обработка {i}/{len(texts)}...", end='\r')
            result = self.extract(text)
            results.append(result)
        print()  # Новая строка после завершения
        return results


def main():
    """Демонстрация работы экстрактора"""
    
    print("🚀 Инициализация NER-системы...")
    
    # Создание экстрактора
    extractor = NameExtractorOllama(model_name="qwen2.5:14b")
    
    # Тестовые примеры
    test_texts = [
        "Меня зовут Иван Петров, я работаю программистом",
        "Документ подписан Анной Сидоровой 15 марта 2024 года",
        "Елена Смирнова отправила письмо вчера вечером",
        "Заявление от Михаила Александровича Иванова о переводе средств",
        "Сергей Николаев и Мария Петрова посетили конференцию",
        "Это текст без имени и фамилии, только информация о погоде",
        "От: Дмитрий Козлов. Тема: Встреча завтра в 10:00",
    ]
    
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ NER-СИСТЕМЫ С OLLAMA SDK")
    print("=" * 80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Пример {i}:")
        print(f"   Исходный текст: \"{text}\"")
        
        try:
            result = extractor.extract(text)
            print(f"\n   ✅ Результат:")
            print(f"      Имя:      {result['first_name'] or 'не найдено'}")
            print(f"      Фамилия:  {result['last_name'] or 'не найдено'}")
            print(f"      Остаток:  {result['remaining_text'] or 'нет'}")
        except Exception as e:
            print(f"   ❌ Ошибка: {str(e)}")
        
        print("   " + "-" * 76)
    
    # Тест пакетной обработки
    print("\n" + "=" * 80)
    print("ПАКЕТНАЯ ОБРАБОТКА")
    print("=" * 80)
    
    batch_texts = test_texts[:3]
    print(f"\nОбработка {len(batch_texts)} текстов...")
    results = extractor.extract_batch(batch_texts)
    
    for i, (text, result) in enumerate(zip(batch_texts, results), 1):
        print(f"\n{i}. {result['first_name']} {result['last_name']}")


if __name__ == "__main__":
    main()

