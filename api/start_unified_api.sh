#!/bin/bash

echo "🚀 Запуск Unified API сервера"
echo "=============================================="
echo ""
echo "📋 Проверка зависимостей..."
echo ""

# Проверка Ollama
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama не найден! Установите: curl -fsSL https://ollama.com/install.sh | sh"
    echo "   NER функционал будет недоступен"
else
    echo "✅ Ollama найден"
    
    # Проверка запущен ли Ollama
    if ! pgrep -x "ollama" > /dev/null; then
        echo "🔄 Запуск Ollama сервера..."
        ollama serve > /dev/null 2>&1 &
        sleep 3
    fi
    
    # Проверка модели qwen2.5:14b
    if ollama list | grep -q "qwen2.5:14b"; then
        echo "✅ Модель qwen2.5:14b найдена"
    else
        echo "⚠️  Модель qwen2.5:14b не найдена"
        echo "   Для установки выполните: ollama pull qwen2.5:14b"
        echo "   NER функционал будет недоступен"
    fi
fi

echo ""
echo "⏳ Загрузка моделей (1-2 минуты)..."
echo ""

cd /home/user/Desktop/RPA/api
python unified_api_server.py




