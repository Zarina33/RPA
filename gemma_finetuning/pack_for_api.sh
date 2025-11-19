#!/bin/bash

echo "📦 Упаковка файлов для развертывания API на другом компьютере"
echo "=============================================================="

# Создаем директорию для упаковки
PACK_DIR="gemma_api_deploy"
rm -rf $PACK_DIR
mkdir -p $PACK_DIR

# Копируем API сервер
echo "📄 Копирование api_server.py..."
cp api_server.py $PACK_DIR/

# Копируем скрипт запуска
echo "📄 Копирование start_api.sh..."
cp start_api.sh $PACK_DIR/

# Копируем словарь
echo "📖 Копирование словаря кодов..."
mkdir -p $PACK_DIR/data
cp data/purpose_codes.txt $PACK_DIR/data/

# Копируем последний checkpoint
echo "💾 Копирование checkpoint..."
LATEST_RUN=$(ls -td outputs/gemma_qlora_* | head -1)
LATEST_CHECKPOINT=$(ls -td $LATEST_RUN/checkpoint-* | head -1)

if [ -d "$LATEST_CHECKPOINT" ]; then
    CHECKPOINT_NAME=$(basename $LATEST_CHECKPOINT)
    RUN_NAME=$(basename $LATEST_RUN)
    
    mkdir -p "$PACK_DIR/outputs/$RUN_NAME"
    cp -r "$LATEST_CHECKPOINT" "$PACK_DIR/outputs/$RUN_NAME/"
    
    echo "   ✅ Checkpoint: $RUN_NAME/$CHECKPOINT_NAME"
else
    echo "   ❌ Checkpoint не найден!"
    exit 1
fi

# Создаем requirements.txt
echo "📝 Создание requirements.txt..."
cat > $PACK_DIR/requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.30.0
bitsandbytes>=0.43.0
peft>=0.11.0
fastapi>=0.110.0
uvicorn>=0.29.0
datasets>=2.19.0
EOF

# Создаем README
echo "📄 Создание README..."
cat > $PACK_DIR/README.txt << 'EOF'
🚀 БЫСТРЫЙ СТАРТ

1. Установка:
   conda create -n gemma_api python=3.10 -y
   conda activate gemma_api
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt

2. Запуск:
   python api_server.py

3. Открыть в браузере:
   http://localhost:8000

Подробная инструкция: см. DEPLOY_API.md
EOF

# Копируем полную документацию
cp DEPLOY_API.md $PACK_DIR/ 2>/dev/null || echo "⚠️  DEPLOY_API.md не найден"
cp API_USAGE.md $PACK_DIR/ 2>/dev/null || echo "⚠️  API_USAGE.md не найден"

# Показываем размер
PACK_SIZE=$(du -sh $PACK_DIR | cut -f1)
echo ""
echo "✅ Упаковка завершена!"
echo "📦 Размер: $PACK_SIZE"
echo "📂 Директория: $PACK_DIR"
echo ""
echo "📋 Содержимое:"
tree -L 3 $PACK_DIR 2>/dev/null || find $PACK_DIR -type f

echo ""
echo "🔄 Для переноса на другой компьютер:"
echo "   1. Создайте архив: tar -czf gemma_api.tar.gz $PACK_DIR"
echo "   2. Скопируйте gemma_api.tar.gz на другой компьютер"
echo "   3. Распакуйте: tar -xzf gemma_api.tar.gz"
echo "   4. Следуйте инструкциям в README.txt"

