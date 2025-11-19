"""
Проверка окружения перед запуском обучения
"""

import sys
import subprocess

def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("   ⚠️  Рекомендуется Python 3.10+")
    else:
        print("   ✅ Версия подходит")
    
    return version.major >= 3 and version.minor >= 10

def check_gpu():
    """Проверка GPU"""
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"\n🎮 CUDA: {'Доступна' if cuda_available else 'Не доступна'}")
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {device_name}")
            print(f"   VRAM: {total_memory:.1f} GB")
            
            if total_memory < 12:
                print(f"   ⚠️  Рекомендуется минимум 12GB VRAM")
                print(f"   💡 У вас {total_memory:.1f}GB - уменьшите BATCH_SIZE")
            elif total_memory < 16:
                print(f"   ⚠️  16GB VRAM оптимально")
            else:
                print(f"   ✅ Достаточно памяти!")
            
            return True
        else:
            print("   ❌ GPU не обнаружена!")
            print("   💡 Обучение на CPU невозможно")
            return False
            
    except ImportError:
        print("\n❌ PyTorch не установлен!")
        return False

def check_dependencies():
    """Проверка установленных библиотек"""
    print("\n📦 Зависимости:")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'bitsandbytes': 'BitsAndBytes',
        'accelerate': 'Accelerate',
        'datasets': 'Datasets',
    }
    
    all_installed = True
    
    for package, name in required.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✅ {name}: {version}")
        except ImportError:
            print(f"   ❌ {name}: не установлен")
            all_installed = False
    
    return all_installed

def check_disk_space():
    """Проверка свободного места на диске"""
    import shutil
    
    print("\n💾 Свободное место:")
    
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    
    print(f"   Свободно: {free_gb:.1f} GB")
    
    if free_gb < 30:
        print(f"   ⚠️  Рекомендуется минимум 30GB")
        print(f"   💡 Модель + датасеты + логи займут ~20-30GB")
        return False
    else:
        print(f"   ✅ Достаточно места")
        return True

def check_data():
    """Проверка наличия исходных данных"""
    from pathlib import Path
    
    print("\n📂 Данные:")
    
    data_file = Path(__file__).parent.parent.parent / 'final_dataset.csv'
    
    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024**2)
        print(f"   ✅ final_dataset.csv найден ({size_mb:.1f} MB)")
        return True
    else:
        print(f"   ❌ final_dataset.csv не найден")
        print(f"   💡 Ожидается: {data_file}")
        return False

def main():
    print("=" * 80)
    print("🔍 ПРОВЕРКА ОКРУЖЕНИЯ ДЛЯ QLORA FINE-TUNING")
    print("=" * 80)
    
    checks = []
    
    # Проверки
    checks.append(("Python", check_python_version()))
    checks.append(("GPU", check_gpu()))
    checks.append(("Зависимости", check_dependencies()))
    checks.append(("Диск", check_disk_space()))
    checks.append(("Данные", check_data()))
    
    # Итог
    print("\n" + "=" * 80)
    print("📊 ИТОГИ ПРОВЕРКИ")
    print("=" * 80)
    
    for name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name}")
    
    all_passed = all(status for _, status in checks)
    
    if all_passed:
        print("\n🎉 Всё готово к запуску обучения!")
        print("\n📝 Следующие шаги:")
        print("   1. python scripts/prepare_data.py")
        print("   2. python scripts/train_qlora.py")
    else:
        print("\n⚠️  Требуются дополнительные настройки")
        print("\n💡 Рекомендации:")
        if not checks[1][1]:  # GPU
            print("   - Установите PyTorch с CUDA")
            print("   - pip install torch --index-url https://download.pytorch.org/whl/cu121")
        if not checks[2][1]:  # Зависимости
            print("   - Установите зависимости: pip install -r requirements.txt")
        if not checks[4][1]:  # Данные
            print("   - Убедитесь, что final_dataset.csv находится в корне проекта")
    
    print("=" * 80)

if __name__ == "__main__":
    main()

