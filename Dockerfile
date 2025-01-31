# Используем официальный Python 3.9 slim образ
FROM python:3.9-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем только requirements.txt для кеширования зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Даем права на выполнение start.sh
RUN chmod +x start.sh

# Указываем команду для запуска приложения
CMD ["./start.sh"]
