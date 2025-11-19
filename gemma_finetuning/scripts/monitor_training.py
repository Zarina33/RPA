"""
Мониторинг обучения в реальном времени
Показывает, все ли идет хорошо
"""

import time
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

# ANSI цвета для терминала
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"

def clear_screen():
    """Очистка экрана"""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_gpu_usage():
    """Получение информации о GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(',')
            return {
                'utilization': int(gpu_util.strip()),
                'memory_used': int(mem_used.strip()),
                'memory_total': int(mem_total.strip()),
                'temperature': int(temp.strip())
            }
    except:
        pass
    return None

def parse_tensorboard_logs(log_dir):
    """Парсинг логов TensorBoard"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()
        
        # Получаем метрики
        metrics = {}
        
        # Loss
        if 'loss' in ea.Tags()['scalars']:
            loss_events = ea.Scalars('loss')
            if loss_events:
                metrics['train_loss'] = loss_events[-1].value
                metrics['train_loss_history'] = [e.value for e in loss_events[-10:]]
        
        # Eval loss
        if 'eval_loss' in ea.Tags()['scalars']:
            eval_events = ea.Scalars('eval_loss')
            if eval_events:
                metrics['eval_loss'] = eval_events[-1].value
        
        # Learning rate
        if 'learning_rate' in ea.Tags()['scalars']:
            lr_events = ea.Scalars('learning_rate')
            if lr_events:
                metrics['learning_rate'] = lr_events[-1].value
        
        return metrics
    except:
        return {}

def read_training_log(log_file):
    """Чтение последних строк лога"""
    try:
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return lines[-20:] if len(lines) > 20 else lines
    except:
        pass
    return []

def analyze_health(gpu_info, metrics, training_time_minutes):
    """Анализ состояния обучения"""
    issues = []
    warnings = []
    good = []
    
    # GPU проверки
    if gpu_info:
        gpu_util = gpu_info['utilization']
        mem_used = gpu_info['memory_used']
        mem_total = gpu_info['memory_total']
        temp = gpu_info['temperature']
        mem_percent = (mem_used / mem_total) * 100
        
        # GPU utilization
        if gpu_util < 30:
            issues.append(f"❌ GPU использование низкое: {gpu_util}% (ожидается >70%)")
        elif gpu_util < 70:
            warnings.append(f"⚠️  GPU использование: {gpu_util}% (можно улучшить)")
        else:
            good.append(f"✅ GPU использование отличное: {gpu_util}%")
        
        # Memory
        if mem_percent > 95:
            warnings.append(f"⚠️  Память GPU почти заполнена: {mem_used}MB / {mem_total}MB ({mem_percent:.1f}%)")
        elif mem_percent > 80:
            good.append(f"✅ Память GPU используется хорошо: {mem_percent:.1f}%")
        else:
            warnings.append(f"⚠️  Память GPU используется мало: {mem_percent:.1f}% (возможно batch size можно увеличить)")
        
        # Temperature
        if temp > 85:
            warnings.append(f"⚠️  Температура GPU высокая: {temp}°C")
        elif temp > 75:
            good.append(f"✅ Температура GPU нормальная: {temp}°C")
        else:
            good.append(f"✅ Температура GPU отличная: {temp}°C")
    
    # Метрики обучения
    if 'train_loss' in metrics:
        train_loss = metrics['train_loss']
        
        if train_loss < 0.5:
            good.append(f"✅ Training loss низкий: {train_loss:.4f} (хорошо!)")
        elif train_loss < 2.0:
            good.append(f"✅ Training loss снижается: {train_loss:.4f}")
        elif train_loss > 5.0 and training_time_minutes > 60:
            warnings.append(f"⚠️  Training loss высокий: {train_loss:.4f} (после 1+ часа)")
        
        # Проверка на застревание loss
        if 'train_loss_history' in metrics and len(metrics['train_loss_history']) > 5:
            history = metrics['train_loss_history']
            if max(history) - min(history) < 0.01:
                warnings.append(f"⚠️  Loss не изменяется последние шаги (возможно застрял)")
    
    if 'eval_loss' in metrics:
        eval_loss = metrics['eval_loss']
        train_loss = metrics.get('train_loss', eval_loss)
        
        # Overfitting check
        if train_loss < eval_loss * 0.7:
            warnings.append(f"⚠️  Возможный overfitting: train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")
        else:
            good.append(f"✅ Eval loss: {eval_loss:.4f}")
    
    if 'learning_rate' in metrics:
        lr = metrics['learning_rate']
        if lr < 1e-6:
            warnings.append(f"⚠️  Learning rate очень маленький: {lr:.2e}")
        else:
            good.append(f"✅ Learning rate: {lr:.2e}")
    
    return issues, warnings, good

def find_latest_run():
    """Поиск последнего запуска обучения"""
    if not LOGS_DIR.exists():
        return None
    
    runs = sorted(LOGS_DIR.glob("gemma_qlora_*"))
    return runs[-1] if runs else None

def estimate_time_remaining(log_dir):
    """Оценка оставшегося времени"""
    try:
        # Читаем checkpoint информацию
        checkpoint_dirs = sorted(Path(log_dir).parent.parent.glob("outputs/gemma_qlora_*/checkpoint-*"))
        if checkpoint_dirs:
            # Получаем номер последнего checkpoint
            last_checkpoint = checkpoint_dirs[-1].name.split('-')[-1]
            return f"Checkpoint: {last_checkpoint}"
    except:
        pass
    return "Неизвестно"

def print_dashboard(gpu_info, metrics, issues, warnings, good, training_time, log_dir):
    """Вывод дашборда"""
    clear_screen()
    
    print("═" * 100)
    print(f"{BOLD}{BLUE}🎯 МОНИТОРИНГ ОБУЧЕНИЯ GEMMA 3:12B{RESET}")
    print("═" * 100)
    
    # Время обучения
    print(f"\n⏱️  {BOLD}Время обучения:{RESET} {training_time}")
    print(f"📁 Лог директория: {log_dir.name if log_dir else 'не найдено'}")
    print(f"🕐 Обновлено: {datetime.now().strftime('%H:%M:%S')}")
    
    # GPU информация
    print(f"\n{BOLD}━━━ 🎮 GPU СТАТУС ━━━{RESET}")
    if gpu_info:
        util = gpu_info['utilization']
        mem_used = gpu_info['memory_used']
        mem_total = gpu_info['memory_total']
        temp = gpu_info['temperature']
        mem_percent = (mem_used / mem_total) * 100
        
        # Progress bars
        util_bar = "█" * (util // 5) + "░" * (20 - util // 5)
        mem_bar = "█" * int(mem_percent // 5) + "░" * (20 - int(mem_percent // 5))
        
        print(f"  Загрузка GPU:  [{util_bar}] {util}%")
        print(f"  Память:        [{mem_bar}] {mem_used} MB / {mem_total} MB ({mem_percent:.1f}%)")
        print(f"  Температура:   {temp}°C")
    else:
        print(f"  {YELLOW}⚠️  Не удалось получить информацию о GPU{RESET}")
    
    # Метрики обучения
    print(f"\n{BOLD}━━━ 📊 МЕТРИКИ ОБУЧЕНИЯ ━━━{RESET}")
    if metrics:
        if 'train_loss' in metrics:
            print(f"  Train Loss:     {metrics['train_loss']:.4f}")
        if 'eval_loss' in metrics:
            print(f"  Eval Loss:      {metrics['eval_loss']:.4f}")
        if 'learning_rate' in metrics:
            print(f"  Learning Rate:  {metrics['learning_rate']:.2e}")
        
        # История loss
        if 'train_loss_history' in metrics and len(metrics['train_loss_history']) > 1:
            history = metrics['train_loss_history']
            trend = "📉 снижается" if history[-1] < history[0] else "📈 растет"
            print(f"  Тренд Loss:     {trend}")
    else:
        print(f"  {YELLOW}⚠️  Метрики пока не доступны (начало обучения){RESET}")
    
    # Диагностика
    print(f"\n{BOLD}━━━ 🏥 ДИАГНОСТИКА ━━━{RESET}")
    
    if issues:
        print(f"\n{RED}{BOLD}❌ ПРОБЛЕМЫ:{RESET}")
        for issue in issues:
            print(f"  {RED}{issue}{RESET}")
    
    if warnings:
        print(f"\n{YELLOW}{BOLD}⚠️  ПРЕДУПРЕЖДЕНИЯ:{RESET}")
        for warning in warnings:
            print(f"  {YELLOW}{warning}{RESET}")
    
    if good:
        print(f"\n{GREEN}{BOLD}✅ ВСЁ ХОРОШО:{RESET}")
        for g in good:
            print(f"  {GREEN}{g}{RESET}")
    
    # Общая оценка
    print(f"\n{BOLD}━━━ 🎯 ОБЩАЯ ОЦЕНКА ━━━{RESET}")
    
    if issues:
        print(f"  {RED}{BOLD}🔴 ТРЕБУЕТСЯ ВНИМАНИЕ!{RESET}")
        print(f"  {RED}Обнаружены критические проблемы. Проверьте логи.{RESET}")
    elif warnings:
        print(f"  {YELLOW}{BOLD}🟡 ОБУЧЕНИЕ ИДЕТ, НО ЕСТЬ ПРЕДУПРЕЖДЕНИЯ{RESET}")
        print(f"  {YELLOW}Можно продолжать, но следите за метриками.{RESET}")
    else:
        print(f"  {GREEN}{BOLD}🟢 ВСЁ ОТЛИЧНО! ОБУЧЕНИЕ ИДЕТ КАК НАДО! 🎉{RESET}")
        print(f"  {GREEN}Продолжайте обучение. Всё под контролем.{RESET}")
    
    # Советы
    print(f"\n{BOLD}━━━ 💡 СОВЕТЫ ━━━{RESET}")
    print(f"  • Для TensorBoard: {BLUE}tensorboard --logdir=logs/{RESET}")
    print(f"  • Для остановки: {RED}Ctrl+C{RESET} (последний checkpoint сохранится)")
    print(f"  • Обновление каждые 10 секунд...")
    
    print("\n" + "═" * 100)

def main():
    """Главная функция мониторинга"""
    print(f"{BOLD}{BLUE}Запуск мониторинга обучения...{RESET}")
    time.sleep(1)
    
    # Находим последний запуск
    log_dir = find_latest_run()
    
    if not log_dir:
        print(f"{RED}❌ Не найдено активных обучений!{RESET}")
        print(f"   Запустите сначала: python scripts/train_qlora.py")
        return
    
    start_time = datetime.now()
    
    print(f"{GREEN}✅ Найдено обучение: {log_dir.name}{RESET}")
    print(f"   Начинаем мониторинг...\n")
    time.sleep(2)
    
    try:
        while True:
            # Время обучения
            elapsed = datetime.now() - start_time
            hours = elapsed.seconds // 3600
            minutes = (elapsed.seconds % 3600) // 60
            training_time = f"{hours}ч {minutes}мин"
            training_time_minutes = elapsed.seconds / 60
            
            # Получаем данные
            gpu_info = get_gpu_usage()
            metrics = parse_tensorboard_logs(log_dir)
            
            # Анализ
            issues, warnings, good = analyze_health(gpu_info, metrics, training_time_minutes)
            
            # Вывод
            print_dashboard(gpu_info, metrics, issues, warnings, good, training_time, log_dir)
            
            # Ждем
            time.sleep(10)
            
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Мониторинг остановлен пользователем.{RESET}")
        print(f"{GREEN}Обучение продолжается в фоне.{RESET}")

if __name__ == "__main__":
    main()

