#!/bin/bash

echo "🚀 Установка NER-системы"
echo "========================"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.8+"
    exit 1
fi

# Проверка Ollama
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama не найден"
    echo "Установить Ollama? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "❌ Установите Ollama вручную: https://ollama.com"
        exit 1
    fi
fi

# Установка Python зависимостей
echo "📦 Установка зависимостей..."
pip install -r requirements.txt

# Проверка запущен ли Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "🔄 Запуск Ollama..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Проверка модели
if ! ollama list | grep -q "qwen2.5:14b"; then
    echo "📥 Загрузить модель qwen2.5:14b? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        ollama pull qwen2.5:14b
    fi
fi

echo ""
echo "✅ Установка завершена!"
echo ""
echo "🚀 Запуск веб-интерфейса:"
echo "   streamlit run ner_web_interface.py"
echo ""
echo "📖 Документация: README.md"

