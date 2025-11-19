# 🚀 Развертывание API на другом компьютере

## 📋 Требования для компьютера:

### Hardware:
- ✅ **GPU:** NVIDIA с минимум 6 GB VRAM (рекомендуется 8+ GB)
- ✅ **CUDA:** 11.8+ или 12.x
- ✅ **RAM:** 8 GB минимум
- ✅ **Диск:** 25-30 GB свободного места

### Software:
- Python 3.10+
- CUDA toolkit
- conda или venv

## 📦 ЧТО НУЖНО СКОПИРОВАТЬ:

```bash
gemma_finetuning/
├── api_server.py              ← API сервер
├── start_api.sh               ← Скрипт запуска
├── outputs/
│   └── gemma_qlora_*/
│       └── checkpoint-XXXX/   ← ЛЮБОЙ checkpoint
├── data/
│   └── purpose_codes.txt      ← Словарь кодов
└── requirements.txt           ← Зависимости
```

## 🔧 УСТАНОВКА НА НОВОМ КОМПЬЮТЕРЕ:

### 1. Создайте окружение:

```bash
# Создать директорию
mkdir -p ~/gemma_api
cd ~/gemma_api

# Создать виртуальное окружение
conda create -n gemma_api python=3.10 -y
conda activate gemma_api

# ИЛИ с venv
python3 -m venv venv
source venv/bin/activate
```

### 2. Установите зависимости:

```bash
# PyTorch с CUDA (важно!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Основные библиотеки
pip install transformers accelerate bitsandbytes peft
pip install fastapi uvicorn datasets

# Проверка CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. Скопируйте файлы:

```bash
# С текущего компьютера на новый (через scp или флешку)
scp -r gemma_finetuning/api_server.py USER@NEW_PC:~/gemma_api/
scp -r gemma_finetuning/outputs/ USER@NEW_PC:~/gemma_api/
scp -r gemma_finetuning/data/purpose_codes.txt USER@NEW_PC:~/gemma_api/data/
```

### 4. Настройте пути в api_server.py:

Откройте `api_server.py` и проверьте пути:

```python
# Строка 17-18
BASE_MODEL = "google/gemma-2-9b-it"
CHECKPOINT_DIR = Path("outputs/gemma_qlora_20251104_181124")  # Ваша папка!
```

## 🚀 ЗАПУСК:

### Простой запуск:

```bash
cd ~/gemma_api
conda activate gemma_api  # или source venv/bin/activate
python api_server.py
```

### Или через uvicorn с настройками:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

### В фоне (daemon):

```bash
nohup python api_server.py > api.log 2>&1 &
```

## 🌐 ДОСТУП К API:

### На локальной машине:
```
http://localhost:8000
```

### С другого компьютера в сети:
```
http://IP_АДРЕС:8000
```

Узнать IP:
```bash
hostname -I
```

## 📊 ИСПОЛЬЗОВАНИЕ VRAM:

API использует **~5-6 GB VRAM** (4-bit квантизация):
- Базовая модель: ~4.5 GB
- LoRA адаптеры: ~0.5 GB
- Inference: ~1 GB

## ⚡ ПРОИЗВОДИТЕЛЬНОСТЬ:

**На GPU:**
- Загрузка модели: 1-2 минуты
- Первый запрос: ~10-15 сек (компиляция CUDA)
- Последующие: **3-5 секунд** ⚡

**На CPU (если нет GPU):**
- Загрузка: 2-3 минуты
- Каждый запрос: 30-60 секунд 🐌

## 🔒 БЕЗОПАСНОСТЬ (для production):

### 1. Добавьте аутентификацию:

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/predict")
async def predict(request: PredictionRequest, token: str = Depends(security)):
    # Проверка токена
    if token != "YOUR_SECRET_TOKEN":
        raise HTTPException(401, "Unauthorized")
    ...
```

### 2. Rate limiting:

```bash
pip install slowapi

from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(...):
    ...
```

### 3. HTTPS (с nginx):

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
    }
}
```

## 🐛 TROUBLESHOOTING:

### "CUDA out of memory"
```python
# Уменьшите batch size в generate
outputs = model.generate(..., batch_size=1)
```

### "Address already in use"
```bash
# Найти и убить процесс
lsof -ti:8000 | xargs kill -9
```

### "Checkpoint not found"
```bash
# Проверьте пути
ls outputs/gemma_qlora_*/checkpoint-*
```

### "Slow first request"
Это нормально - CUDA kernels компилируются при первом запуске.

## 📝 АЛЬТЕРНАТИВНЫЕ ЧЕКПОИНТЫ:

API автоматически использует **последний checkpoint**.

Для выбора конкретного, измените в `api_server.py`:

```python
# Строка 20-22
LATEST_CHECKPOINT = Path("outputs/gemma_qlora_20251104_181124/checkpoint-7000")
```

Доступные checkpoints:
- **checkpoint-1000**: epoch 0.06 (~5% quality)
- **checkpoint-3000**: epoch 0.18 (~15% quality)
- **checkpoint-7000**: epoch 0.40 (~35-40% quality) ← рекомендуется!
- **checkpoint-17000**: epoch 1.0 (~65% quality)

## 🎯 РЕКОМЕНДАЦИИ:

1. **Для тестирования:** используйте checkpoint-3000
2. **Для демо:** checkpoint-7000+
3. **Для production:** дождитесь окончания обучения (3 эпохи)

---

**Вопросы?** Проверьте логи: `tail -f api.log`

