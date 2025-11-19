#!/bin/bash
# Быстрый старт для QLoRA Fine-tuning Gemma 3:12b

set -e  # Остановка при ошибках

echo "========================================"
echo "🚀 QUICKSTART: QLoRA Fine-tuning Gemma"
echo "========================================"

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Проверка окружения
echo ""
log_info "Шаг 1: Проверка окружения..."
python scripts/check_environment.py

if [ $? -ne 0 ]; then
    log_error "Проверка окружения не пройдена!"
    exit 1
fi

# 2. Подготовка данных
echo ""
log_info "Шаг 2: Подготовка данных..."
if [ ! -f "data/train.jsonl" ]; then
    python scripts/prepare_data.py
    log_info "✅ Данные подготовлены"
else
    log_warn "Данные уже подготовлены. Пропускаем..."
fi

# 3. Запуск обучения
echo ""
log_info "Шаг 3: Запуск обучения..."
log_warn "⏱️  Это займет 1-2 дня на RTX 3060 Ti"
echo ""
read -p "Начать обучение? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    log_info "Запуск обучения..."
    python scripts/train_qlora.py
    
    if [ $? -eq 0 ]; then
        log_info "🎉 Обучение завершено успешно!"
    else
        log_error "Обучение завершилось с ошибкой"
        exit 1
    fi
else
    log_warn "Обучение отменено пользователем"
    exit 0
fi

# 4. Evaluation
echo ""
log_info "Шаг 4: Evaluation модели..."
read -p "Запустить evaluation? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python scripts/evaluate.py
    log_info "✅ Evaluation завершен"
fi

# 5. Inference demo
echo ""
log_info "Шаг 5: Тестирование inference..."
read -p "Запустить демо inference? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python scripts/inference.py
fi

echo ""
echo "========================================"
log_info "🎉 QUICKSTART ЗАВЕРШЕН!"
echo "========================================"

