"""
API для тестирования fine-tuned модели
Запуск: uvicorn api_server:app --host 0.0.0.0 --port 8000
"""
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json

# Настройки
BASE_MODEL = "google/gemma-2-9b-it"
CHECKPOINT_DIR = Path("outputs/gemma_qlora_20251104_181124")

# Находим последний checkpoint
checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
LATEST_CHECKPOINT = checkpoints[-1] if checkpoints else None

print("=" * 80)
print("🚀 ЗАПУСК API СЕРВЕРА")
print("=" * 80)

if LATEST_CHECKPOINT:
    print(f"📂 Используется checkpoint: {LATEST_CHECKPOINT.name}")
else:
    print("❌ Не найдено checkpoints!")
    exit(1)

# Загрузка словаря кодов
DICTIONARY = {}
try:
    with open('../data/purpose_codes.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('=') and not line.startswith('CreditCode') and not line.startswith('---') and 'Всего записей' not in line and 'СЛОВАРЬ' not in line:
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    DICTIONARY[parts[0]] = parts[1].strip()
    print(f"📖 Загружено кодов в словаре: {len(DICTIONARY)}")
except:
    print("⚠️  Словарь не загружен")

# Загрузка модели
print("\n📥 Загрузка модели...")
print("   (это займет 1-2 минуты)")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Настройка 4-bit квантизации для экономии VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
) if torch.cuda.is_available() else None

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",  # Автоматически использует GPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, LATEST_CHECKPOINT)
model.eval()

print("   ✅ Модель загружена и готова!\n")

# FastAPI app
app = FastAPI(
    title="OperCode Prediction API",
    description="API для предсказания кодов операций банковских платежей",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    payment_comment: str
    temperature: float = 0.1
    max_tokens: int = 10

class PredictionResponse(BaseModel):
    input_text: str
    predicted_code: str
    code_description: str
    model_response: str
    confidence: str
    checkpoint: str

def extract_code(text):
    """Извлекает код операции из ответа модели"""
    numbers = re.findall(r'\b\d{5,6}\b', text)  # Ищем 5-6 значные числа
    if numbers:
        return numbers[0]
    # Запасной вариант - любое число
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[0]
    return None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Веб-интерфейс для тестирования"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OperCode Predictor</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 900px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .checkpoint-info {
                background: #f0f7ff;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 20px;
                text-align: center;
                font-size: 14px;
                color: #555;
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                resize: vertical;
                min-height: 120px;
                box-sizing: border-box;
            }
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                margin-top: 15px;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                display: none;
            }
            .result.show {
                display: block;
            }
            .result-item {
                margin: 15px 0;
                padding: 15px;
                background: white;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .result-label {
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }
            .result-value {
                color: #333;
                font-size: 18px;
            }
            .code-result {
                font-size: 32px;
                font-weight: bold;
                color: #667eea;
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #f0f7ff 0%, #e6f0ff 100%);
                border-radius: 8px;
                margin: 20px 0;
            }
            .loading {
                display: none;
                text-align: center;
                color: #667eea;
                margin-top: 15px;
            }
            .loading.show {
                display: block;
            }
            .examples {
                margin-top: 20px;
                padding: 15px;
                background: #fff3cd;
                border-radius: 8px;
            }
            .example {
                cursor: pointer;
                padding: 8px;
                margin: 5px 0;
                background: white;
                border-radius: 5px;
                border: 1px solid #ffc107;
                transition: background 0.2s;
            }
            .example:hover {
                background: #fff8e1;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏦 OperCode Predictor</h1>
            <div class="subtitle">Определение кода операции банковского платежа</div>
            
            <div class="checkpoint-info">
                📊 Checkpoint: """ + LATEST_CHECKPOINT.name + """ | 🧠 Model: Gemma 2 9B + QLoRA
            </div>
            
            <div>
                <textarea id="paymentInput" placeholder="Введите описание банковского платежа...
Например: PURCHASE OF GOODS FROM COMPANY ABC"></textarea>
                
                <button onclick="predict()" id="predictBtn">🔮 Определить код операции</button>
                
                <div class="loading" id="loading">
                    ⏳ Модель анализирует... (5-10 секунд)
                </div>
            </div>
            
            <div class="result" id="result">
                <div class="code-result" id="codeResult"></div>
                
                <div class="result-item">
                    <div class="result-label">📝 Описание операции:</div>
                    <div class="result-value" id="description"></div>
                </div>
            </div>
            
            <div class="examples">
                <strong>💡 Примеры для тестирования:</strong>
                <div class="example" onclick="setExample('PURCHASE OF GOODS')">
                    PURCHASE OF GOODS
                </div>
                <div class="example" onclick="setExample('TRANSFER OF FUNDS TO OWN ACCOUNT')">
                    TRANSFER OF FUNDS TO OWN ACCOUNT
                </div>
                <div class="example" onclick="setExample('Переводы между физическими лицами')">
                    Переводы между физическими лицами
                </div>
                <div class="example" onclick="setExample('Оплата за услуги консультирования')">
                    Оплата за услуги консультирования
                </div>
            </div>
        </div>
        
        <script>
            function setExample(text) {
                document.getElementById('paymentInput').value = text;
            }
            
            async function predict() {
                const input = document.getElementById('paymentInput').value.trim();
                
                if (!input) {
                    alert('Пожалуйста, введите описание платежа');
                    return;
                }
                
                // UI updates
                document.getElementById('predictBtn').disabled = true;
                document.getElementById('loading').classList.add('show');
                document.getElementById('result').classList.remove('show');
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({payment_comment: input})
                    });
                    
                    const data = await response.json();
                    
                    // Display results
                    document.getElementById('codeResult').textContent = 
                        '🎯 Код: ' + data.predicted_code;
                    document.getElementById('description').textContent = 
                        data.code_description;
                    
                    document.getElementById('result').classList.add('show');
                    
                } catch (error) {
                    alert('Ошибка: ' + error.message);
                } finally {
                    document.getElementById('predictBtn').disabled = false;
                    document.getElementById('loading').classList.remove('show');
                }
            }
            
            // Enter to submit
            document.getElementById('paymentInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) {
                    predict();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Предсказание кода операции"""
    try:
        # Создаем промпт
        prompt = f"""<start_of_turn>user
Определи код операции (OperCode) для следующего банковского платежа:

Платёж: {request.payment_comment}

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
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
            )
        
        # Декодирование
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_response = response.split("<start_of_turn>model")[-1].strip()
        
        # Извлечение кода
        predicted_code = extract_code(model_response)
        
        if not predicted_code:
            predicted_code = "НЕ ОПРЕДЕЛЕН"
            code_description = "Модель не смогла определить код"
            confidence = "Низкая"
        else:
            code_description = DICTIONARY.get(predicted_code, "⚠️ Код не найден в словаре")
            confidence = "Средняя" if len(model_response) > 10 else "Высокая"
        
        return PredictionResponse(
            input_text=request.payment_comment,
            predicted_code=predicted_code,
            code_description=code_description,
            model_response=model_response[:200],
            confidence=confidence,
            checkpoint=LATEST_CHECKPOINT.name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Проверка работоспособности"""
    return {
        "status": "ok",
        "model": BASE_MODEL,
        "checkpoint": LATEST_CHECKPOINT.name,
        "device": str(model.device)
    }

@app.get("/codes")
async def get_codes():
    """Получить список всех кодов"""
    return {
        "total": len(DICTIONARY),
        "codes": DICTIONARY
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 80)
    print("🌐 Сервер запущен!")
    print("=" * 80)
    print("📱 Веб-интерфейс: http://localhost:8000")
    print("📡 API docs: http://localhost:8000/docs")
    print("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8000)

