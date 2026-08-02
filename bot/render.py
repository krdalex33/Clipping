"""Рендер одного вертикального клипа 9:16 — ОДИН вызов ffmpeg.

Порядок фильтров критичен и соответствует ТЗ:
  1. Вырезка (-ss/-t, с перекодированием).
  2. Уникализация геометрии: hflip, затем лёгкий зум ~1.03. ЗЕРКАЛО ДО СУБТИТРОВ.
  3. Цветокор: eq + hue.
  4. Вертикаль 1080x1920 с размытым фоном (split/blur/overlay).
  5. Субтитры: ass= (файл передаём коротким относительным именем из cwd).
  6. Ускорение: setpts=PTS/1.04 (видео) + atempo=1.04 (аудио) — скорости совпадают.
"""
from __future__ import annotations

import logging
import os

from . import config
from .ffmpeg_utils import FFmpegError, probe_summary, run_ffmpeg
from .subtitles import build_ass
from .transcribe import Word

log = logging.getLogger("clipping.render")

MB = 1024 * 1024
SIZE_LIMIT = 45 * MB          # порог пересжатия
TELEGRAM_VIDEO_LIMIT = 50 * MB  # выше — только документом


class RenderError(RuntimeError):
    pass


def _video_filter(fonts_dir: str) -> str:
    # Один общий видеофильтр. subs.ass — относительное имя (cwd=workdir).
    return (
        "[0:v]hflip,"
        "scale=iw*1.03:ih*1.03,crop=iw/1.03:ih/1.03,"
        "eq=brightness=0.02:saturation=1.06,hue=h=2,"
        "split[a][b];"
        "[a]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=20:2[bg];"
        "[b]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
        f"ass=subs.ass:fontsdir={fonts_dir},"
        "setpts=PTS/1.04[v]"
    )


def _encode(
    video_path: str,
    start: float,
    length: float,
    workdir: str,
    out_name: str,
    fonts_dir: str,
    crf: int,
    has_audio: bool,
) -> str:
    """Собрать и запустить ffmpeg. cwd=workdir, чтобы subs.ass читался коротко."""
    filt = _video_filter(fonts_dir)

    # Вырезаем на входе: -ss (быстрый seek) и -t ДО -i, чтобы читались ровно
    # `length` секунд от `start`. Так однозначно и для видео без аудиодорожки.
    args: list[str] = [
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
        "-i", os.path.abspath(video_path),
    ]
    tail: list[str] = []

    if has_audio:
        audio_filter = "[0:a]atempo=1.04[a]"
        filter_complex = f"{filt};{audio_filter}"
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        # Нет аудиодорожки — подставляем тишину, чтобы клип всегда был со звуком.
        args += ["-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=44100:cl=stereo"]
        filter_complex = filt
        maps = ["-map", "[v]", "-map", "1:a"]
        tail = ["-shortest"]  # длина по видео, а не по бесконечной тишине

    args += [
        "-filter_complex", filter_complex,
        *maps,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-map_metadata", "-1",
        "-r", "30",
        *tail,
        out_name,
    ]
    run_ffmpeg(args, cwd=workdir)
    out_path = os.path.join(workdir, out_name)
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RenderError("ffmpeg отработал, но клип пустой.")
    return out_path


def render_clip(
    video_path: str,
    clip: dict,
    words: list[Word],
    workdir: str,
    index: int,
) -> dict:
    """Отрендерить один клип. Вернуть dict со сводкой.

    clip: {"start", "end", "reason"}. Возвращает:
      {path, size_bytes, duration, width, height, has_audio, subs_lines,
       as_document(bool), reason}
    """
    fonts_dir = config.FONTS_DIR
    start = float(clip["start"])
    end = float(clip["end"])
    length = max(1.0, end - start)

    # Есть ли в исходнике аудио.
    try:
        src = probe_summary(video_path)
        has_audio = bool(src.get("has_audio"))
    except FFmpegError:
        has_audio = True

    # Субтитры для окна клипа.
    subs_path = os.path.join(workdir, "subs.ass")
    subs_lines = build_ass(words, start, end, subs_path)

    out_name = f"clip_{index:02d}.mp4"
    out_path = _encode(video_path, start, length, workdir, out_name,
                        fonts_dir, crf=24, has_audio=has_audio)

    size = os.path.getsize(out_path)
    # Пересжатие, если крупный.
    if size > SIZE_LIMIT:
        log.info("Клип %d = %.1f МБ > 45 МБ — пересжимаю crf=28.", index, size / MB)
        out_path = _encode(video_path, start, length, workdir, out_name,
                           fonts_dir, crf=28, has_audio=has_audio)
        size = os.path.getsize(out_path)

    summary = probe_summary(out_path)
    as_document = size > TELEGRAM_VIDEO_LIMIT
    if as_document:
        log.warning("Клип %d = %.1f МБ > 50 МБ — уйдёт документом.", index, size / MB)

    return {
        "path": out_path,
        "size_bytes": size,
        "duration": summary.get("duration", length),
        "width": summary.get("width"),
        "height": summary.get("height"),
        "has_audio": summary.get("has_audio", False),
        "subs_lines": subs_lines,
        "as_document": as_document,
        "reason": clip.get("reason", ""),
    }
