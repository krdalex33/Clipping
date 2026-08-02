"""Единая обёртка над ffmpeg/ffprobe.

Каждый вызов — через subprocess с capture_output, проверкой returncode и
логированием последних 2000 символов stderr. Никаких тихих падений.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

from . import config

log = logging.getLogger("clipping.ffmpeg")

STDERR_TAIL = 2000


class FFmpegError(RuntimeError):
    """Ошибка выполнения ffmpeg/ffprobe с хвостом stderr внутри."""


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str


def run_ffmpeg(args: list[str], *, cwd: str | None = None, timeout: int = 1800) -> RunResult:
    """Запустить ffmpeg. args — БЕЗ имени бинарника.

    Кидает FFmpegError с последними 2000 символами stderr при ненулевом коде.
    """
    cmd = [config.FFMPEG_BIN, "-hide_banner", "-nostdin", "-y", *args]
    log.info("ffmpeg: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffmpeg завис (>{timeout}с) и был убит.") from exc
    except FileNotFoundError as exc:
        raise FFmpegError(f"Не найден бинарник {config.FFMPEG_BIN}.") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "")[-STDERR_TAIL:]
        log.error("ffmpeg упал (code=%d). stderr(tail):\n%s", proc.returncode, tail)
        raise FFmpegError(
            f"ffmpeg вернул код {proc.returncode}. Последние {STDERR_TAIL} символов stderr:\n{tail}"
        )
    return RunResult(proc.returncode, proc.stdout or "", proc.stderr or "")


def ffprobe_json(path: str, *, timeout: int = 120) -> dict:
    """Вернуть метаданные файла (format + streams) как dict."""
    cmd = [
        config.FFPROBE_BIN, "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffprobe завис (>{timeout}с).") from exc
    except FileNotFoundError as exc:
        raise FFmpegError(f"Не найден бинарник {config.FFPROBE_BIN}.") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "")[-STDERR_TAIL:]
        raise FFmpegError(f"ffprobe вернул код {proc.returncode}:\n{tail}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe вернул не-JSON: {proc.stdout[:500]}") from exc


def get_duration(path: str) -> float:
    """Длительность файла в секундах (0.0 если не удалось определить)."""
    info = ffprobe_json(path)
    dur = info.get("format", {}).get("duration")
    if dur is not None:
        try:
            return float(dur)
        except (TypeError, ValueError):
            pass
    # запасной вариант — по видеопотоку
    for st in info.get("streams", []):
        if st.get("duration"):
            try:
                return float(st["duration"])
            except (TypeError, ValueError):
                continue
    return 0.0


def probe_summary(path: str) -> dict:
    """Компактная сводка о медиафайле для отчётов/логов."""
    info = ffprobe_json(path)
    fmt = info.get("format", {})
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    summary = {
        "duration": float(fmt.get("duration", 0) or 0),
        "size_bytes": int(fmt.get("size", 0) or 0),
        "has_video": video is not None,
        "has_audio": audio is not None,
        "width": video.get("width") if video else None,
        "height": video.get("height") if video else None,
        "v_codec": video.get("codec_name") if video else None,
        "a_codec": audio.get("codec_name") if audio else None,
    }
    return summary
