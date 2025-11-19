"""
Unified API для OperCode Prediction + NER (извлечение имен)
Запуск: python unified_api_server.py
"""
import torch
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json

# Добавляем путь к NER модулю
sys.path.append(str(Path(__file__).parent.parent / "NER"))
from ner_extraction_ollama import NameExtractorOllama

# ============================================================================
# НАСТРОЙКИ
# ============================================================================
BASE_MODEL = "google/gemma-2-9b-it"
CHECKPOINT_DIR = Path(__file__).parent.parent / "gemma_finetuning/outputs/gemma_qlora_20251104_181124"

# Находим последний checkpoint
checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
LATEST_CHECKPOINT = checkpoints[-1] if checkpoints else None

print("=" * 80)
print("🚀 ЗАПУСК UNIFIED API СЕРВЕРА")
print("=" * 80)

if LATEST_CHECKPOINT:
    print(f"📂 Используется checkpoint: {LATEST_CHECKPOINT.name}")
else:
    print("❌ Не найдено checkpoints!")
    exit(1)

# ============================================================================
# ЗАГРУЗКА СЛОВАРЯ КОДОВ
# ============================================================================
DICTIONARY = {}
try:
    purpose_codes_path = Path(__file__).parent.parent / "data/purpose_codes.txt"
    with open(purpose_codes_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('=') and not line.startswith('CreditCode') and not line.startswith('---') and 'Всего записей' not in line and 'СЛОВАРЬ' not in line:
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    DICTIONARY[parts[0]] = parts[1].strip()
    print(f"📖 Загружено кодов в словаре: {len(DICTIONARY)}")
except:
    print("⚠️  Словарь не загружен")

# ============================================================================
# ЗАГРУЗКА GEMMA МОДЕЛИ
# ============================================================================
print("\n📥 Загрузка Gemma модели для OperCode...")
print("   (это займет 1-2 минуты)")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
) if torch.cuda.is_available() else None

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, LATEST_CHECKPOINT)
model.eval()

print("   ✅ Gemma модель загружена!\n")

# ============================================================================
# ЗАГРУЗКА NER МОДЕЛИ
# ============================================================================
print("📥 Инициализация NER системы...")
try:
    ner_extractor = NameExtractorOllama(model_name="qwen2.5:14b")
    print("   ✅ NER система готова!\n")
except Exception as e:
    print(f"   ⚠️  NER система недоступна: {e}\n")
    ner_extractor = None

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Unified Banking API",
    description="API для предсказания кодов операций и извлечения имен",
    version="2.0.0"
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class OperCodeRequest(BaseModel):
    payment_comment: str
    temperature: float = 0.1
    max_tokens: int = 10

class OperCodeResponse(BaseModel):
    input_text: str
    predicted_code: str
    code_description: str
    confidence: str
    checkpoint: str

class NERRequest(BaseModel):
    text: str

class NERResponse(BaseModel):
    first_name: str = None
    last_name: str = None
    full_name: str = None
    remaining_text: str = ""

class UnifiedRequest(BaseModel):
    text: str
    extract_opercode: bool = True
    extract_names: bool = True

class UnifiedResponse(BaseModel):
    input_text: str
    opercode: dict = None
    ner: dict = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extract_code(text):
    """Извлекает код операции из ответа модели"""
    numbers = re.findall(r'\b\d{5,6}\b', text)
    if numbers:
        return numbers[0]
    numbers = re.findall(r'\d+', text)
    if numbers:
        return numbers[0]
    return None

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с объединенным интерфейсом"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Banking Analysis Suite</title>
        <meta charset="utf-8">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            .tabs {
                display: flex;
                background: #f8f9fa;
                border-bottom: 2px solid #dee2e6;
            }
            .tab {
                flex: 1;
                padding: 20px;
                text-align: center;
                cursor: pointer;
                font-size: 1.1rem;
                font-weight: 600;
                color: #666;
                transition: all 0.3s;
                border-bottom: 3px solid transparent;
            }
            .tab:hover {
                background: #e9ecef;
            }
            .tab.active {
                color: #667eea;
                background: white;
                border-bottom-color: #667eea;
            }
            .tab-content {
                display: none;
                padding: 30px;
            }
            .tab-content.active {
                display: block;
            }
            .input-group {
                margin-bottom: 20px;
            }
            .input-group label {
                display: block;
                font-weight: 600;
                margin-bottom: 10px;
                color: #333;
                font-size: 1.1rem;
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 16px;
                resize: vertical;
                min-height: 120px;
                font-family: inherit;
                transition: border-color 0.3s;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            .button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
            }
            .button:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            .result {
                margin-top: 30px;
                padding: 25px;
                background: #f8f9fa;
                border-radius: 10px;
                display: none;
            }
            .result.show {
                display: block;
            }
            .result-section {
                margin-bottom: 25px;
                padding: 20px;
                background: white;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            .result-section h3 {
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.3rem;
            }
            .result-item {
                margin: 10px 0;
                display: flex;
                align-items: center;
            }
            .result-label {
                font-weight: 600;
                color: #666;
                min-width: 150px;
            }
            .result-value {
                color: #333;
                font-size: 1.1rem;
                font-weight: 500;
            }
            .code-badge {
                display: inline-block;
                padding: 10px 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 25px;
                font-size: 1.5rem;
                font-weight: bold;
            }
            .name-badge {
                display: inline-block;
                padding: 8px 16px;
                background: #28a745;
                color: white;
                border-radius: 20px;
                font-size: 1.2rem;
                margin-right: 10px;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
                color: #667eea;
                font-size: 1.1rem;
            }
            .loading.show {
                display: block;
            }
            .examples {
                margin-top: 20px;
                padding: 20px;
                background: #fff3cd;
                border-radius: 10px;
                border: 2px solid #ffc107;
            }
            .examples h4 {
                color: #856404;
                margin-bottom: 15px;
            }
            .example {
                cursor: pointer;
                padding: 12px;
                margin: 8px 0;
                background: white;
                border-radius: 8px;
                border: 1px solid #ffc107;
                transition: all 0.2s;
            }
            .example:hover {
                background: #fff8e1;
                transform: translateX(5px);
            }
            .checkbox-group {
                margin: 20px 0;
                padding: 15px;
                background: #e7f3ff;
                border-radius: 10px;
            }
            .checkbox-label {
                display: flex;
                align-items: center;
                margin: 10px 0;
                cursor: pointer;
            }
            .checkbox-label input {
                margin-right: 10px;
                width: 20px;
                height: 20px;
                cursor: pointer;
            }
            .info-box {
                padding: 15px;
                background: #d1ecf1;
                border-left: 4px solid #17a2b8;
                border-radius: 5px;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏦 Banking Analysis Suite</h1>
                <p>Определение кодов операций и извлечение имен из текста</p>
            </div>
            
            <div class="tabs">
                <div class="tab active" onclick="switchTab('opercode')">
                    💳 OperCode Predictor
                </div>
                <div class="tab" onclick="switchTab('ner')">
                    👤 Name Extractor
                </div>
                <div class="tab" onclick="switchTab('unified')">
                    🔄 Unified Analysis
                </div>
            </div>
            
            <!-- OperCode Tab -->
            <div id="opercode-tab" class="tab-content active">
                <div class="info-box">
                    <strong>📊 Checkpoint:</strong> """ + LATEST_CHECKPOINT.name + """ | 
                    <strong>🧠 Model:</strong> Gemma 2 9B + QLoRA
                </div>
                
                <div class="input-group">
                    <label>Введите описание банковского платежа:</label>
                    <textarea id="opercodeInput" placeholder="Например: PURCHASE OF GOODS FROM COMPANY ABC"></textarea>
                </div>
                
                <button class="button" onclick="predictOperCode()">🔮 Определить код операции</button>
                
                <div class="loading" id="opercodeLoading">
                    ⏳ Модель анализирует... (5-10 секунд)
                </div>
                
                <div class="result" id="opercodeResult">
                    <div class="result-section">
                        <h3>🎯 Результат</h3>
                        <div class="result-item">
                            <span class="result-label">Код операции:</span>
                            <span class="code-badge" id="codeValue"></span>
                        </div>
                        <div class="result-item">
                            <span class="result-label">Описание:</span>
                            <span class="result-value" id="codeDescription"></span>
                        </div>
                    </div>
                </div>
                
                <div class="examples">
                    <h4>💡 Примеры для тестирования:</h4>
                    <div class="example" onclick="setExample('opercode', 'PURCHASE OF GOODS')">
                        PURCHASE OF GOODS
                    </div>
                    <div class="example" onclick="setExample('opercode', 'TRANSFER OF FUNDS TO OWN ACCOUNT')">
                        TRANSFER OF FUNDS TO OWN ACCOUNT
                    </div>
                    <div class="example" onclick="setExample('opercode', 'Переводы между физическими лицами')">
                        Переводы между физическими лицами
                    </div>
                    <div class="example" onclick="setExample('opercode', 'Оплата за услуги консультирования')">
                        Оплата за услуги консультирования
                    </div>
                </div>
            </div>
            
            <!-- NER Tab -->
            <div id="ner-tab" class="tab-content">
                <div class="info-box">
                    <strong>🤖 Model:</strong> Qwen 2.5 14B via Ollama | 
                    <strong>🎯 Task:</strong> Named Entity Recognition
                </div>
                
                <div class="input-group">
                    <label>Введите текст для извлечения имени и фамилии:</label>
                    <textarea id="nerInput" placeholder="Например: Меня зовут Иван Петров, я работаю программистом"></textarea>
                </div>
                
                <button class="button" onclick="extractNames()">👤 Извлечь имя и фамилию</button>
                
                <div class="loading" id="nerLoading">
                    ⏳ Анализ текста... (5-10 секунд)
                </div>
                
                <div class="result" id="nerResult">
                    <div class="result-section">
                        <h3>👤 Извлеченные данные</h3>
                        <div class="result-item">
                            <span class="result-label">Полное имя:</span>
                            <span id="fullName"></span>
                        </div>
                        <div class="result-item">
                            <span class="result-label">Имя:</span>
                            <span class="result-value" id="firstName"></span>
                        </div>
                        <div class="result-item">
                            <span class="result-label">Фамилия:</span>
                            <span class="result-value" id="lastName"></span>
                        </div>
                        <div class="result-item" id="remainingTextBlock" style="display:none; margin-top: 15px;">
                            <span class="result-label">Остальной текст:</span>
                            <span class="result-value" id="remainingText"></span>
                        </div>
                    </div>
                </div>
                
                <div class="examples">
                    <h4>💡 Примеры для тестирования:</h4>
                    <div class="example" onclick="setExample('ner', 'Меня зовут Иван Петров, я работаю программистом')">
                        Меня зовут Иван Петров, я работаю программистом
                    </div>
                    <div class="example" onclick="setExample('ner', 'Документ подписан Анной Сидоровой 15 марта 2024 года')">
                        Документ подписан Анной Сидоровой 15 марта 2024 года
                    </div>
                    <div class="example" onclick="setExample('ner', 'От: Дмитрий Козлов. Тема: Встреча завтра в 10:00')">
                        От: Дмитрий Козлов. Тема: Встреча завтра в 10:00
                    </div>
                </div>
            </div>
            
            <!-- Unified Tab -->
            <div id="unified-tab" class="tab-content">
                <div class="info-box">
                    <strong>🔄 Unified Analysis:</strong> Анализ текста обеими моделями одновременно
                </div>
                
                <div class="input-group">
                    <label>Введите текст для комплексного анализа:</label>
                    <textarea id="unifiedInput" placeholder="Например: Иван Петров совершил перевод средств на свой счет"></textarea>
                </div>
                
                <div class="checkbox-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="checkOperCode" checked>
                        <span>Определить код операции</span>
                    </label>
                    <label class="checkbox-label">
                        <input type="checkbox" id="checkNER" checked>
                        <span>Извлечь имя и фамилию</span>
                    </label>
                </div>
                
                <button class="button" onclick="unifiedAnalysis()">🔄 Выполнить анализ</button>
                
                <div class="loading" id="unifiedLoading">
                    ⏳ Комплексный анализ... (10-20 секунд)
                </div>
                
                <div class="result" id="unifiedResult">
                    <div class="result-section" id="unifiedOperCodeSection" style="display:none;">
                        <h3>💳 Код операции</h3>
                        <div class="result-item">
                            <span class="result-label">Код:</span>
                            <span class="code-badge" id="unifiedCode"></span>
                        </div>
                        <div class="result-item">
                            <span class="result-label">Описание:</span>
                            <span class="result-value" id="unifiedCodeDesc"></span>
                        </div>
                    </div>
                    
                    <div class="result-section" id="unifiedNERSection" style="display:none;">
                        <h3>👤 Извлеченные имена</h3>
                        <div class="result-item">
                            <span class="result-label">Полное имя:</span>
                            <span id="unifiedFullName"></span>
                        </div>
                        <div class="result-item">
                            <span class="result-label">Имя:</span>
                            <span class="result-value" id="unifiedFirstName"></span>
                        </div>
                        <div class="result-item">
                            <span class="result-label">Фамилия:</span>
                            <span class="result-value" id="unifiedLastName"></span>
                        </div>
                    </div>
                </div>
                
                <div class="examples">
                    <h4>💡 Примеры для тестирования:</h4>
                    <div class="example" onclick="setExample('unified', 'Иван Петров совершил перевод средств на свой счет')">
                        Иван Петров совершил перевод средств на свой счет
                    </div>
                    <div class="example" onclick="setExample('unified', 'Анна Сидорова оплатила покупку товаров')">
                        Анна Сидорова оплатила покупку товаров
                    </div>
                    <div class="example" onclick="setExample('unified', 'Михаил Козлов - оплата консультационных услуг')">
                        Михаил Козлов - оплата консультационных услуг
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function switchTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                
                // Show selected tab
                document.getElementById(tabName + '-tab').classList.add('active');
                event.target.classList.add('active');
            }
            
            function setExample(tab, text) {
                if (tab === 'opercode') {
                    document.getElementById('opercodeInput').value = text;
                } else if (tab === 'ner') {
                    document.getElementById('nerInput').value = text;
                } else if (tab === 'unified') {
                    document.getElementById('unifiedInput').value = text;
                }
            }
            
            async function predictOperCode() {
                const input = document.getElementById('opercodeInput').value.trim();
                if (!input) {
                    alert('Пожалуйста, введите описание платежа');
                    return;
                }
                
                document.getElementById('opercodeLoading').classList.add('show');
                document.getElementById('opercodeResult').classList.remove('show');
                
                try {
                    const response = await fetch('/predict/opercode', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({payment_comment: input})
                    });
                    
                    const data = await response.json();
                    
                    document.getElementById('codeValue').textContent = data.predicted_code;
                    document.getElementById('codeDescription').textContent = data.code_description;
                    document.getElementById('opercodeResult').classList.add('show');
                    
                } catch (error) {
                    alert('Ошибка: ' + error.message);
                } finally {
                    document.getElementById('opercodeLoading').classList.remove('show');
                }
            }
            
            async function extractNames() {
                const input = document.getElementById('nerInput').value.trim();
                if (!input) {
                    alert('Пожалуйста, введите текст');
                    return;
                }
                
                document.getElementById('nerLoading').classList.add('show');
                document.getElementById('nerResult').classList.remove('show');
                
                try {
                    const response = await fetch('/predict/ner', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: input})
                    });
                    
                    const data = await response.json();
                    
                    const firstName = data.first_name || 'не найдено';
                    const lastName = data.last_name || 'не найдено';
                    
                    document.getElementById('firstName').textContent = firstName;
                    document.getElementById('lastName').textContent = lastName;
                    
                    if (data.full_name) {
                        document.getElementById('fullName').innerHTML = 
                            '<span class="name-badge">' + data.full_name + '</span>';
                    } else {
                        document.getElementById('fullName').textContent = 'не найдено';
                    }
                    
                    if (data.remaining_text) {
                        document.getElementById('remainingText').textContent = data.remaining_text;
                        document.getElementById('remainingTextBlock').style.display = 'flex';
                    } else {
                        document.getElementById('remainingTextBlock').style.display = 'none';
                    }
                    
                    document.getElementById('nerResult').classList.add('show');
                    
                } catch (error) {
                    alert('Ошибка: ' + error.message);
                } finally {
                    document.getElementById('nerLoading').classList.remove('show');
                }
            }
            
            async function unifiedAnalysis() {
                const input = document.getElementById('unifiedInput').value.trim();
                if (!input) {
                    alert('Пожалуйста, введите текст');
                    return;
                }
                
                const extractOperCode = document.getElementById('checkOperCode').checked;
                const extractNER = document.getElementById('checkNER').checked;
                
                if (!extractOperCode && !extractNER) {
                    alert('Выберите хотя бы один тип анализа');
                    return;
                }
                
                document.getElementById('unifiedLoading').classList.add('show');
                document.getElementById('unifiedResult').classList.remove('show');
                
                try {
                    const response = await fetch('/predict/unified', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            text: input,
                            extract_opercode: extractOperCode,
                            extract_names: extractNER
                        })
                    });
                    
                    const data = await response.json();
                    
                    // OperCode results
                    if (data.opercode) {
                        document.getElementById('unifiedCode').textContent = data.opercode.predicted_code;
                        document.getElementById('unifiedCodeDesc').textContent = data.opercode.code_description;
                        document.getElementById('unifiedOperCodeSection').style.display = 'block';
                    } else {
                        document.getElementById('unifiedOperCodeSection').style.display = 'none';
                    }
                    
                    // NER results
                    if (data.ner) {
                        const firstName = data.ner.first_name || 'не найдено';
                        const lastName = data.ner.last_name || 'не найдено';
                        
                        document.getElementById('unifiedFirstName').textContent = firstName;
                        document.getElementById('unifiedLastName').textContent = lastName;
                        
                        if (data.ner.full_name) {
                            document.getElementById('unifiedFullName').innerHTML = 
                                '<span class="name-badge">' + data.ner.full_name + '</span>';
                        } else {
                            document.getElementById('unifiedFullName').textContent = 'не найдено';
                        }
                        
                        document.getElementById('unifiedNERSection').style.display = 'block';
                    } else {
                        document.getElementById('unifiedNERSection').style.display = 'none';
                    }
                    
                    document.getElementById('unifiedResult').classList.add('show');
                    
                } catch (error) {
                    alert('Ошибка: ' + error.message);
                } finally {
                    document.getElementById('unifiedLoading').classList.remove('show');
                }
            }
            
            // Ctrl+Enter to submit
            document.getElementById('opercodeInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) predictOperCode();
            });
            document.getElementById('nerInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) extractNames();
            });
            document.getElementById('unifiedInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) unifiedAnalysis();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/opercode", response_model=OperCodeResponse)
async def predict_opercode(request: OperCodeRequest):
    """Предсказание кода операции"""
    try:
        prompt = f"""<start_of_turn>user
Определи код операции (OperCode) для следующего банковского платежа:

Платёж: {request.payment_comment}

Ответь только числовым кодом операции.<end_of_turn>
<start_of_turn>model
"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_response = response.split("<start_of_turn>model")[-1].strip()
        
        predicted_code = extract_code(model_response)
        
        if not predicted_code:
            predicted_code = "НЕ ОПРЕДЕЛЕН"
            code_description = "Модель не смогла определить код"
            confidence = "Низкая"
        else:
            code_description = DICTIONARY.get(predicted_code, "⚠️ Код не найден в словаре")
            confidence = "Средняя" if len(model_response) > 10 else "Высокая"
        
        return OperCodeResponse(
            input_text=request.payment_comment,
            predicted_code=predicted_code,
            code_description=code_description,
            confidence=confidence,
            checkpoint=LATEST_CHECKPOINT.name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/ner", response_model=NERResponse)
async def predict_ner(request: NERRequest):
    """Извлечение имени и фамилии"""
    if ner_extractor is None:
        raise HTTPException(status_code=503, detail="NER система недоступна. Убедитесь, что Ollama запущен.")
    
    try:
        result = ner_extractor.extract(request.text)
        
        full_name = None
        if result['first_name'] and result['last_name']:
            full_name = f"{result['first_name']} {result['last_name']}"
        
        return NERResponse(
            first_name=result['first_name'],
            last_name=result['last_name'],
            full_name=full_name,
            remaining_text=result['remaining_text']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/unified", response_model=UnifiedResponse)
async def predict_unified(request: UnifiedRequest):
    """Комплексный анализ: OperCode + NER"""
    result = UnifiedResponse(input_text=request.text)
    
    # OperCode prediction
    if request.extract_opercode:
        try:
            opercode_req = OperCodeRequest(payment_comment=request.text)
            opercode_result = await predict_opercode(opercode_req)
            result.opercode = {
                "predicted_code": opercode_result.predicted_code,
                "code_description": opercode_result.code_description,
                "confidence": opercode_result.confidence
            }
        except Exception as e:
            result.opercode = {"error": str(e)}
    
    # NER extraction
    if request.extract_names:
        try:
            ner_req = NERRequest(text=request.text)
            ner_result = await predict_ner(ner_req)
            result.ner = {
                "first_name": ner_result.first_name,
                "last_name": ner_result.last_name,
                "full_name": ner_result.full_name,
                "remaining_text": ner_result.remaining_text
            }
        except Exception as e:
            result.ner = {"error": str(e)}
    
    return result

@app.get("/health")
async def health():
    """Проверка работоспособности"""
    return {
        "status": "ok",
        "gemma_model": BASE_MODEL,
        "checkpoint": LATEST_CHECKPOINT.name,
        "device": str(model.device),
        "ner_available": ner_extractor is not None
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
    print("🌐 UNIFIED API СЕРВЕР ЗАПУЩЕН!")
    print("=" * 80)
    print("📱 Веб-интерфейс: http://localhost:8000")
    print("📡 API docs: http://localhost:8000/docs")
    print("=" * 80)
    print("\n💡 Доступные функции:")
    print("   - OperCode Prediction (Gemma 2 9B)")
    print("   - Name Extraction (Qwen 2.5 14B)")
    print("   - Unified Analysis (оба одновременно)")
    print("=" * 80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)




