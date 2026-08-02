FROM python:3.12-slim

# Системные зависимости: ffmpeg (монтаж), шрифты с кириллицей, корневые сертификаты.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — так слой кэшируется и пересборка быстрее.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Затем весь проект (код, шрифты assets/fonts и т.д.).
COPY . .

# Небуферизованный вывод — логи сразу видны в панели Bothost.
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "bot.main"]
