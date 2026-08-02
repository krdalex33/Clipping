"""Оркестрация всего пайплайна.

download -> audio -> transcribe -> select -> render (по одному, отправка сразу).
В finally рабочая папка задачи удаляется целиком.

Тяжёлые (блокирующие) операции уводим в отдельный поток через asyncio.to_thread,
чтобы бот не подвисал.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from typing import Awaitable, Callable

from . import audio, downloader, render, selector, transcribe
from .config import Config
from .downloader import DownloadError

log = logging.getLogger("clipping.pipeline")

StatusCb = Callable[[str], Awaitable[None]]
ClipCb = Callable[[dict], Awaitable[None]]

WORK_BASE = os.environ.get("WORK_BASE", tempfile.gettempdir())


class PipelineError(RuntimeError):
    """Ошибка с готовым текстом для пользователя."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


async def process(
    url: str,
    cfg: Config,
    on_status: StatusCb,
    on_clip: ClipCb,
) -> int:
    """Обработать одну ссылку. Возвращает число отправленных клипов.

    on_status(text)  — прислать пользователю статус.
    on_clip(summary) — прислать готовый клип (summary из render.render_clip).
    """
    workdir = tempfile.mkdtemp(prefix="clip_", dir=WORK_BASE)
    sent = 0
    try:
        # 1. Скачивание
        await on_status("⏬ Скачиваю видео…")
        try:
            dl = await asyncio.to_thread(downloader.download, url, workdir, cfg)
        except DownloadError as exc:
            raise PipelineError(exc.user_message) from exc

        # 2. Аудио
        await on_status("🎧 Извлекаю аудио…")
        audio_path = await asyncio.to_thread(audio.extract_audio, dl.path, workdir)
        chunks = await asyncio.to_thread(audio.split_if_needed, audio_path, workdir)

        # 3. Транскрипция
        await on_status("📝 Распознаю речь (Groq Whisper)…")
        try:
            tr = await asyncio.to_thread(transcribe.transcribe, chunks, cfg.groq_api_key)
        except transcribe.TranscribeError as exc:
            raise PipelineError(exc.user_message) from exc

        # 4. Выбор моментов
        await on_status(f"🧠 Выбираю моменты (язык: {tr.language})…")
        try:
            clips = await asyncio.to_thread(
                selector.select_clips, tr, dl.duration, cfg.groq_api_key, cfg.clips_count
            )
        except selector.SelectError as exc:
            raise PipelineError(exc.user_message) from exc

        if not clips:
            raise PipelineError("Не нашлось подходящих фрагментов в этом видео.")

        await on_status(f"🎬 Готовлю {len(clips)} клип(ов), пришлю по мере готовности…")

        # 5. Рендер по одному + немедленная отправка
        for i, clip in enumerate(clips, start=1):
            try:
                summary = await asyncio.to_thread(
                    render.render_clip, dl.path, clip, tr.words, workdir, i
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("Ошибка рендера клипа %d", i)
                await on_status(f"⚠️ Клип {i} не удалось собрать: {exc}. Иду дальше.")
                continue
            await on_clip(summary)
            sent += 1

        if sent == 0:
            raise PipelineError("Ни один клип не удалось собрать. Попробуй другое видео.")
        return sent
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        log.info("Рабочая папка удалена: %s", workdir)
