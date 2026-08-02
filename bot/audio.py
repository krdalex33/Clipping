"""Извлечение аудио для транскрипции.

- mono, 16 кГц, mp3 64k;
- если результат > 20 МБ — режем на куски по 10 минут (для лимита Groq по размеру).
"""
from __future__ import annotations

import logging
import os

from .ffmpeg_utils import get_duration, run_ffmpeg

log = logging.getLogger("clipping.audio")

MAX_AUDIO_BYTES = 20 * 1024 * 1024  # 20 МБ
CHUNK_SECONDS = 10 * 60  # 10 минут


def extract_audio(video_path: str, workdir: str) -> str:
    """Извлечь аудио в mono/16k/mp3-64k. Вернуть путь к mp3."""
    out = os.path.join(workdir, "audio.mp3")
    run_ffmpeg([
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        "-f", "mp3",
        out,
    ])
    if not os.path.isfile(out):
        raise RuntimeError("Аудио не извлеклось (файл не создан).")
    log.info("Аудио: %s (%d байт)", out, os.path.getsize(out))
    return out


def split_if_needed(audio_path: str, workdir: str) -> list[tuple[str, float]]:
    """Вернуть список (путь_к_куску, offset_секунд).

    Если файл влезает в лимит — один кусок с offset=0.
    Иначе режем на куски по CHUNK_SECONDS без перекодирования (-c copy).
    """
    size = os.path.getsize(audio_path)
    if size <= MAX_AUDIO_BYTES:
        return [(audio_path, 0.0)]

    total = get_duration(audio_path)
    log.info("Аудио %d байт > лимита — режем на куски по %d c (всего %.0f c).",
             size, CHUNK_SECONDS, total)

    chunks: list[tuple[str, float]] = []
    idx = 0
    start = 0.0
    while start < total:
        chunk_path = os.path.join(workdir, f"audio_chunk_{idx:03d}.mp3")
        run_ffmpeg([
            "-ss", f"{start:.3f}",
            "-i", audio_path,
            "-t", f"{CHUNK_SECONDS}",
            "-c", "copy",
            chunk_path,
        ])
        if os.path.isfile(chunk_path) and os.path.getsize(chunk_path) > 0:
            chunks.append((chunk_path, start))
        idx += 1
        start += CHUNK_SECONDS

    if not chunks:
        raise RuntimeError("Не удалось нарезать аудио на куски.")
    return chunks
