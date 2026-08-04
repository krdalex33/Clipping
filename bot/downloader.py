"""Скачивание видео с YouTube через yt-dlp (as a library).

Задачи:
- формат не выше 720p;
- отказ, если исходник длиннее 90 минут;
- поддержка cookies и прокси (обход блокировок);
- разбор типичных ошибок блокировки в человеческий текст.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import yt_dlp

from .config import Config

log = logging.getLogger("clipping.downloader")

MAX_SOURCE_SECONDS = 90 * 60  # 90 минут
FORMAT = "bv*[height<=720]+ba/b[height<=720]/b"

# Максимум времени на само скачивание. YouTube часто режет скорость для серверов
# хостинга; без этого лимита бот висел бы на одной задаче бесконечно.
DOWNLOAD_TIMEOUT = 8 * 60  # 8 минут
_TIMEOUT_MARK = "clipping-download-timeout"


class _Deadline:
    """progress-hook для yt-dlp: прерывает слишком долгое скачивание."""

    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + seconds

    def __call__(self, status: dict) -> None:
        if time.monotonic() > self.deadline:
            raise yt_dlp.utils.DownloadError(_TIMEOUT_MARK)


class DownloadError(RuntimeError):
    """Ошибка скачивания с готовым текстом для пользователя."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class SourceTooLong(DownloadError):
    pass


@dataclass
class DownloadResult:
    path: str
    title: str
    duration: float


def _base_opts(cfg: Config, workdir: str) -> dict:
    opts: dict = {
        "format": FORMAT,
        "outtmpl": os.path.join(workdir, "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 5,
        # Обрываем зависшее соединение вместо бесконечного ожидания.
        "socket_timeout": 30,
        # Качаем фрагменты параллельно — заметно быстрее на DASH-форматах.
        "concurrent_fragment_downloads": 4,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "noprogress": True,
        # ВАЖНО: используем cookie-совместимые клиенты. iOS/Android-клиенты быстрее,
        # но ИГНОРИРУЮТ cookies, из-за чего с серверного IP YouTube отдаёт
        # «видео недоступно». web/mweb/tv работают с нашими cookies.
        "extractor_args": {"youtube": {"player_client": ["web", "mweb", "tv"]}},
        # Немного маскируемся под обычный клиент.
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    }
    if cfg.cookies_file and os.path.isfile(cfg.cookies_file):
        opts["cookiefile"] = cfg.cookies_file
    if cfg.yt_proxy:
        opts["proxy"] = cfg.yt_proxy
    return opts


def _humanize(err_text: str) -> str:
    low = err_text.lower()
    if _TIMEOUT_MARK in err_text or "timed out" in low or "timeout" in low:
        return (
            "Видео качается слишком медленно — YouTube режет скорость для серверов "
            "хостинга. Что можно сделать:\n"
            "• пришли ролик покороче (5–10 минут);\n"
            "• обнови cookies (YT_COOKIES_B64);\n"
            "• для стабильной скорости задай прокси в переменной YT_PROXY "
            "(лучше резидентный)."
        )
    if any(k in low for k in ("sign in to confirm", "not a bot", "confirm you", "cookies")):
        return (
            "YouTube требует авторизацию (антибот). Обнови cookies:\n"
            "1) залогинься на youtube.com в браузере;\n"
            "2) выгрузи cookies.txt (см. README, раздел «Cookies»);\n"
            "3) сделай base64 и обнови переменную YT_COOKIES_B64 на Bothost;\n"
            "4) перезапусти бота."
        )
    if "age" in low and ("restrict" in low or "confirm" in low):
        return ("Видео с возрастным ограничением — нужны cookies залогиненного "
                "аккаунта. Обнови YT_COOKIES_B64 (см. README).")
    if any(k in low for k in ("private video", "video unavailable", "removed", "not available")):
        return "Видео недоступно (приватное, удалено или заблокировано в регионе)."
    if "unsupported url" in low or "is not a valid url" in low:
        return "Это не похоже на ссылку YouTube. Пришли обычную ссылку на видео."
    if "http error 429" in low or "too many requests" in low:
        return ("YouTube временно ограничил запросы (429). Подожди 10–15 минут "
                "или задай прокси через YT_PROXY.")
    return ("Не удалось скачать видео. Возможные причины: устаревшие cookies, "
            "блокировка YouTube или недоступность видео. Технические детали в логах.")


def download(url: str, workdir: str, cfg: Config) -> DownloadResult:
    """Скачать видео в workdir. Бросает DownloadError с готовым текстом."""
    opts = _base_opts(cfg, workdir)

    # Шаг 1. Узнаём длительность до полной загрузки, чтобы отсеять длинные.
    try:
        with yt_dlp.YoutubeDL({**opts, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(_humanize(str(exc))) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Непредвиденная ошибка при extract_info")
        raise DownloadError(_humanize(str(exc))) from exc

    if info is None:
        raise DownloadError(_humanize("video unavailable"))

    duration = float(info.get("duration") or 0)
    title = info.get("title") or "video"
    if duration and duration > MAX_SOURCE_SECONDS:
        mins = int(duration // 60)
        raise SourceTooLong(
            f"Видео слишком длинное ({mins} мин). Максимум — 90 минут. "
            f"Пришли ролик покороче."
        )

    # Шаг 2. Собственно скачивание — с дедлайном, чтобы не висеть вечно.
    dl_opts = {**opts, "progress_hooks": [_Deadline(DOWNLOAD_TIMEOUT)]}
    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(_humanize(str(exc))) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Непредвиденная ошибка при скачивании")
        raise DownloadError(_humanize(str(exc))) from exc

    # После merge расширение может стать .mp4 — ищем реальный файл.
    if not os.path.isfile(path):
        root, _ = os.path.splitext(path)
        for ext in (".mp4", ".mkv", ".webm"):
            cand = root + ext
            if os.path.isfile(cand):
                path = cand
                break
    if not os.path.isfile(path):
        raise DownloadError("Файл скачался, но не найден на диске — попробуй ещё раз.")

    log.info("Скачано: %s (%.1f c) -> %s", title, duration, path)
    return DownloadResult(path=path, title=title, duration=duration or 0.0)
