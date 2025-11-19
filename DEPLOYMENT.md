# 🚀 Deployment Guide

Руководство по развертыванию Banking Analysis Suite на production сервере.

## 📋 Требования к серверу

### Hardware
- **GPU:** NVIDIA с 8+ GB VRAM (рекомендуется 12+ GB)
- **RAM:** 16+ GB (рекомендуется 32 GB)
- **CPU:** 4+ cores
- **Диск:** 50+ GB SSD
- **Сеть:** Стабильное интернет-соединение

### Software
- **OS:** Ubuntu 20.04+ / Debian 11+
- **Python:** 3.10+
- **CUDA:** 11.8+ или 12.x
- **Docker:** (опционально)

## 🔧 Установка на сервер

### 1. Подготовка сервера

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка зависимостей
sudo apt install -y build-essential git curl wget
sudo apt install -y python3.10 python3.10-venv python3-pip

# Установка NVIDIA драйверов (если еще не установлены)
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 2. Установка CUDA

```bash
# CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# Добавьте в ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Клонирование репозитория

```bash
cd /opt
sudo git clone https://github.com/YOUR_USERNAME/RPA.git
sudo chown -R $USER:$USER /opt/RPA
cd /opt/RPA
```

### 4. Создание виртуального окружения

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 5. Установка зависимостей

```bash
# PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Остальные зависимости
pip install -r requirements.txt
```

### 6. Установка Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Загрузка модели
ollama pull qwen2.5:14b
```

### 7. Настройка Hugging Face

```bash
# Авторизация в Hugging Face
pip install huggingface-hub
huggingface-cli login

# Запрос доступа к Gemma
# Перейдите на https://huggingface.co/google/gemma-2-9b-it
# и запросите доступ
```

## 🔐 Настройка безопасности

### 1. Создание пользователя для сервиса

```bash
sudo useradd -r -s /bin/false rpa-api
sudo chown -R rpa-api:rpa-api /opt/RPA
```

### 2. Настройка firewall

```bash
# UFW
sudo ufw allow 8000/tcp
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
```

### 3. SSL/TLS с Nginx

```bash
# Установка Nginx
sudo apt install -y nginx certbot python3-certbot-nginx

# Конфигурация
sudo nano /etc/nginx/sites-available/rpa-api
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts для длинных запросов
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
        send_timeout 600;
    }
}
```

```bash
# Активация конфигурации
sudo ln -s /etc/nginx/sites-available/rpa-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# SSL сертификат
sudo certbot --nginx -d your-domain.com
```

## 🔄 Systemd сервисы

### 1. Ollama сервис

```bash
sudo nano /etc/systemd/system/ollama.service
```

```ini
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=rpa-api
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 2. API сервис

```bash
sudo nano /etc/systemd/system/rpa-api.service
```

```ini
[Unit]
Description=RPA Banking Analysis API
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=rpa-api
WorkingDirectory=/opt/RPA/api
Environment="PATH=/opt/RPA/venv/bin"
ExecStart=/opt/RPA/venv/bin/python unified_api_server.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/rpa-api/output.log
StandardError=append:/var/log/rpa-api/error.log

[Install]
WantedBy=multi-user.target
```

### 3. Создание директории для логов

```bash
sudo mkdir -p /var/log/rpa-api
sudo chown rpa-api:rpa-api /var/log/rpa-api
```

### 4. Запуск сервисов

```bash
# Перезагрузка systemd
sudo systemctl daemon-reload

# Запуск сервисов
sudo systemctl start ollama
sudo systemctl start rpa-api

# Автозапуск
sudo systemctl enable ollama
sudo systemctl enable rpa-api

# Проверка статуса
sudo systemctl status ollama
sudo systemctl status rpa-api
```

## 📊 Мониторинг

### 1. Логи

```bash
# Логи API
sudo journalctl -u rpa-api -f

# Логи Ollama
sudo journalctl -u ollama -f

# Логи Nginx
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 2. Мониторинг GPU

```bash
# Установка nvidia-smi
watch -n 1 nvidia-smi

# Или используйте gpustat
pip install gpustat
watch -n 1 gpustat
```

### 3. Мониторинг системы

```bash
# htop
sudo apt install htop
htop

# Использование диска
df -h

# Использование памяти
free -h
```

## 🔄 Обновление

```bash
cd /opt/RPA
git pull origin main

# Активация окружения
source venv/bin/activate

# Обновление зависимостей
pip install -r requirements.txt --upgrade

# Перезапуск сервиса
sudo systemctl restart rpa-api
```

## 💾 Backup

### Что нужно бэкапить:

1. **Checkpoints модели:**
   ```bash
   /opt/RPA/gemma_finetuning/outputs/
   ```

2. **Данные:**
   ```bash
   /opt/RPA/data/
   ```

3. **Конфигурация:**
   ```bash
   /opt/RPA/api/
   /etc/nginx/sites-available/rpa-api
   /etc/systemd/system/rpa-api.service
   ```

### Скрипт backup:

```bash
#!/bin/bash
BACKUP_DIR="/backup/rpa-$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup checkpoints
tar -czf $BACKUP_DIR/checkpoints.tar.gz /opt/RPA/gemma_finetuning/outputs/

# Backup data
tar -czf $BACKUP_DIR/data.tar.gz /opt/RPA/data/

# Backup config
cp /etc/nginx/sites-available/rpa-api $BACKUP_DIR/
cp /etc/systemd/system/rpa-api.service $BACKUP_DIR/

echo "Backup completed: $BACKUP_DIR"
```

## 🐛 Troubleshooting

### API не запускается

```bash
# Проверка логов
sudo journalctl -u rpa-api -n 100

# Проверка портов
sudo lsof -i :8000

# Проверка GPU
nvidia-smi
```

### Ollama недоступен

```bash
# Проверка статуса
sudo systemctl status ollama

# Перезапуск
sudo systemctl restart ollama

# Проверка модели
ollama list
```

### Высокое использование памяти

```bash
# Мониторинг
free -h
nvidia-smi

# Перезапуск сервиса
sudo systemctl restart rpa-api
```

## 📈 Масштабирование

### Horizontal scaling

Для обработки большего количества запросов:

1. Используйте load balancer (nginx, HAProxy)
2. Запустите несколько инстансов API на разных серверах
3. Используйте Redis для кэширования

### Vertical scaling

- Увеличьте RAM
- Используйте более мощную GPU
- Используйте SSD для быстрого доступа к моделям

---

**Production ready!** 🚀

