"""Чтение и валидация переменных окружения.

Секреты и настройки берём ОДИН раз при старте. Если чего-то критичного нет —
падаем сразу с понятным сообщением, а не посреди обработки видео.
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import sys
from dataclasses import dataclass

log = logging.getLogger("clipping.config")

# Пути внутри контейнера (WORKDIR /app).
COOKIES_PATH = "/app/cookies.txt"
FONTS_DIR = os.environ.get("FONTS_DIR", "/app/assets/fonts")

# Позволяем переопределить бинарники ffmpeg/ffprobe (нужно для локального теста,
# на Bothost они системные и оставляем значения по умолчанию).
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")


@dataclass
class Config:
    bot_token: str
    groq_api_key: str
    yt_proxy: str | None
    admin_id: int | None
    clips_count: int
    cookies_file: str | None  # путь к cookies.txt, если удалось раскодировать


def _decode_cookies() -> str | None:
    """YT_COOKIES_B64 -> /app/cookies.txt. Возвращает путь или None."""
    b64 = os.environ.get("YT_COOKIES_B64", "").strip()
    if not b64:
        log.warning("YT_COOKIES_B64 не задан — YouTube может требовать авторизацию.")
        return None
    try:
        raw = base64.b64decode(b64, validate=False)
        # Пытаемся записать в /app/cookies.txt; если каталог недоступен (локальный
        # запуск не в контейнере) — кладём рядом во временный файл.
        target = COOKIES_PATH
        try:
            with open(target, "wb") as fh:
                fh.write(raw)
        except OSError:
            target = os.path.join(os.getcwd(), "cookies.txt")
            with open(target, "wb") as fh:
                fh.write(raw)
        if len(raw) < 20:
            log.warning("cookies.txt подозрительно маленький (%d байт).", len(raw))
        log.info("cookies.txt записан: %s (%d байт).", target, len(raw))
        return target
    except Exception as exc:  # noqa: BLE001 — хотим любой сбой превратить в предупреждение
        log.error("Не удалось раскодировать YT_COOKIES_B64: %s. Работаем без cookies.", exc)
        return None


def _require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        log.critical("Не задана обязательная переменная окружения %s.", name)
        sys.exit(f"ОШИБКА: не задана переменная окружения {name}. "
                 f"Заполните её в настройках Bothost (или в .env).")
    return val


def _binary_present(binary: str) -> bool:
    return shutil.which(binary) is not None or os.path.isfile(binary)


def _try_static_ffmpeg() -> None:
    """Если системного ffmpeg нет (хостинг собрал не через Dockerfile) —
    подтягиваем статические ffmpeg+ffprobe пакетом static-ffmpeg и добавляем в PATH.
    Скачивание идёт один раз при первом старте."""
    try:
        import static_ffmpeg  # noqa: PLC0415 — импортируем лениво, только при нужде
        log.info("Системный ffmpeg не найден — готовлю статический через static-ffmpeg…")
        static_ffmpeg.add_paths()  # скачает при необходимости и допишет PATH
        log.info("static-ffmpeg готов, ffmpeg/ffprobe добавлены в PATH.")
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось подготовить static-ffmpeg: %s", exc)


def check_ffmpeg() -> None:
    """Убеждаемся, что ffmpeg и ffprobe доступны. Иначе — падаем с инструкцией."""
    if not (_binary_present(FFMPEG_BIN) and _binary_present(FFPROBE_BIN)):
        _try_static_ffmpeg()

    for binary in (FFMPEG_BIN, FFPROBE_BIN):
        if not _binary_present(binary):
            log.critical("Не найден %s.", binary)
            sys.exit(
                f"ОШИБКА: не найден {binary}. Ожидался системный ffmpeg (Dockerfile: "
                f"'apt-get install -y ffmpeg') или пакет static-ffmpeg из requirements.txt."
            )
    log.info("ffmpeg и ffprobe на месте.")


def load_config() -> Config:
    """Собираем конфиг. Вызывать один раз при старте."""
    bot_token = _require("BOT_TOKEN")
    groq_api_key = _require("GROQ_API_KEY")

    yt_proxy = os.environ.get("YT_PROXY", "").strip() or None

    admin_raw = os.environ.get("ADMIN_ID", "").strip()
    admin_id: int | None = None
    if admin_raw:
        try:
            admin_id = int(admin_raw)
        except ValueError:
            log.warning("ADMIN_ID='%s' не число — игнорирую.", admin_raw)

    clips_raw = os.environ.get("CLIPS_COUNT", "3").strip()
    try:
        clips_count = int(clips_raw)
    except ValueError:
        clips_count = 3
    clips_count = max(1, min(5, clips_count))  # держим в разумных рамках 1..5

    cookies_file = _decode_cookies()

    if not os.path.isdir(FONTS_DIR):
        log.warning("Каталог шрифтов %s не найден — субтитры могут остаться без шрифта.", FONTS_DIR)

    log.info(
        "Конфиг загружен: clips_count=%d, proxy=%s, cookies=%s, admin=%s",
        clips_count, bool(yt_proxy), bool(cookies_file), admin_id,
    )
    return Config(
        bot_token=bot_token,
        groq_api_key=groq_api_key,
        yt_proxy=yt_proxy,
        admin_id=admin_id,
        clips_count=clips_count,
        cookies_file=cookies_file,
    )
