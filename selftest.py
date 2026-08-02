#!/usr/bin/env python3
"""Самотестирование пайплайна БЕЗ Telegram.

Прогоняет: получение исходного видео -> извлечение аудио -> (Groq) транскрипция
-> (Groq) выбор моментов -> рендер клипов, и печатает человекочитаемый отчёт
с реальными цифрами (длительность, разрешение, размер, звук, субтитры).

Источник видео (по приоритету):
  1. переменная окружения TEST_VIDEO — путь к готовому видеофайлу;
  2. macOS `say` — синтез реальной речи (для локальной проверки);
  3. синтетическое видео ffmpeg (тон + движущийся кадр) — речи нет.

Транскрипция/выбор:
  • если задан GROQ_API_KEY и в аудио есть речь — работает по-настоящему;
  • иначе используется встроенный «фейковый» транскрипт, а шаги помечаются в отчёте.

Коды выхода: 0 — успех, 1 — провал.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot import audio as audio_mod  # noqa: E402
from bot import config, render, selector, transcribe  # noqa: E402
from bot.ffmpeg_utils import FFmpegError, probe_summary, run_ffmpeg  # noqa: E402
from bot.transcribe import Transcript, Word  # noqa: E402

MB = 1024 * 1024

# Русский монолог для синтеза речи (macOS `say`) — законченные мысли.
SPEECH_TEXT = (
    "Привет! Сегодня я расскажу, как делать короткие вертикальные видео. "
    "Первое правило простое: сильное начало решает всё. "
    "Если первые три секунды скучные, зритель сразу уходит. "
    "Поэтому начинайте с самого яркого момента, а не с долгого вступления. "
    "Второе правило: одна мысль — один клип. Не пытайтесь уместить всё сразу. "
    "Третье: субтитры обязательны, ведь большинство смотрит без звука. "
    "И самое главное — публикуйте регулярно, тогда алгоритм вас полюбит. "
    "Давайте поговорим подробнее про удержание внимания зрителя. "
    "Хороший клип держит ритм: короткие фразы, паузы в нужных местах, живая интонация. "
    "Не бойтесь показывать эмоции, ведь именно они заставляют досматривать до конца. "
    "Экспериментируйте с обложками и первыми словами, тестируйте разные заходы. "
    "Смотрите, какие клипы залетают, и повторяйте удачные приёмы снова и снова. "
    "Спасибо за внимание, до встречи в следующем видео!"
)


def _p(msg: str = "") -> None:
    print(msg, flush=True)


def _has(binary: str) -> bool:
    return shutil.which(binary) is not None


def make_source_with_say(workdir: str) -> str | None:
    """Синтез речи через macOS `say` -> видео со звуком. None, если не вышло."""
    if not _has("say"):
        return None
    aiff = os.path.join(workdir, "speech.aiff")
    # Пытаемся русским голосом; если нет — голосом по умолчанию.
    voices = ["Milena", "Yuri", None]
    ok = False
    for v in voices:
        cmd = ["say"]
        if v:
            cmd += ["-v", v]
        cmd += ["-o", aiff, SPEECH_TEXT]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.isfile(aiff) and os.path.getsize(aiff) > 0:
            ok = True
            _p(f"  speech: голос={v or 'по умолчанию'}")
            break
    if not ok:
        return None

    # Определяем длительность речи.
    from bot.ffmpeg_utils import get_duration
    dur = max(30.0, get_duration(aiff))
    src = os.path.join(workdir, "source.mp4")
    # Видео: цветной фон + бегущая надпись + дорожка речи.
    run_ffmpeg([
        "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={dur:.2f}",
        "-i", aiff,
        "-filter_complex",
        "[0:v]drawtext=text='SELFTEST':fontcolor=white:fontsize=90:"
        "x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.5:boxborderw=20[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        src,
    ])
    return src


def make_source_synthetic(workdir: str) -> str:
    """Синтетическое видео (тон + testsrc), без речи."""
    src = os.path.join(workdir, "source.mp4")
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=50",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=50",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        src,
    ])
    return src


def mock_transcript(duration: float) -> Transcript:
    """Фейковый транскрипт: покрывает всё видео короткими словами по ~0.45 c."""
    sample = ("это тестовый клип с субтитрами проверяем шрифт кириллицу "
              "жирный текст обводку и позицию снизу кадра всё работает").split()
    words: list[Word] = []
    t = 0.5
    i = 0
    while t < duration - 0.5:
        w = sample[i % len(sample)]
        words.append(Word(word=w, start=round(t, 2), end=round(t + 0.4, 2)))
        t += 0.45
        i += 1
    return Transcript(language="ru (mock)", words=words, text=" ".join(w.word for w in words))


def naive_clips(duration: float, count: int) -> list[dict]:
    """Простой выбор окон по 35 c, если LLM недоступен."""
    clips: list[dict] = []
    length = 35.0 if duration >= 40 else max(5.0, duration - 2)
    start = 1.0
    for i in range(count):
        end = min(start + length, duration - 0.5)
        if end - start < 4:
            break
        clips.append({"start": round(start, 2), "end": round(end, 2),
                      "reason": f"Автовыбор окна #{i+1} (без LLM)."})
        start = end + 1
        if start >= duration - 5:
            break
    if not clips:
        clips = [{"start": 0.5, "end": min(duration - 0.5, 20.0),
                  "reason": "Единственное доступное окно."}]
    return clips


def main() -> int:
    _p("=" * 64)
    _p("SELFTEST: пайплайн нарезки клипов")
    _p("=" * 64)

    # 0. ffmpeg
    try:
        config.check_ffmpeg()
    except SystemExit as e:
        _p(f"[FAIL] ffmpeg: {e}")
        return 1
    _p(f"[ok] ffmpeg: {config.FFMPEG_BIN}, ffprobe: {config.FFPROBE_BIN}")

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    _p(f"[info] GROQ_API_KEY: {'задан' if groq_key else 'НЕ задан (Groq-шаги будут пропущены)'}")

    workdir = tempfile.mkdtemp(prefix="selftest_")
    _p(f"[info] рабочая папка: {workdir}")
    keep = "--keep" in sys.argv

    report: list[str] = []
    ok_all = True
    try:
        # 1. Источник
        _p("\n[1/5] Готовлю исходное видео…")
        env_video = os.environ.get("TEST_VIDEO", "").strip()
        speech_source = False
        if env_video and os.path.isfile(env_video):
            src = env_video
            _p(f"  использую TEST_VIDEO: {src}")
        else:
            src = make_source_with_say(workdir)
            if src:
                speech_source = True
                _p("  источник: синтез речи (macOS say)")
            else:
                src = make_source_synthetic(workdir)
                _p("  источник: синтетическое видео (без речи)")
        s = probe_summary(src)
        _p(f"  исходник: {s['width']}x{s['height']}, {s['duration']:.1f} c, "
           f"звук={'да' if s['has_audio'] else 'нет'}")

        # 2. Аудио
        _p("\n[2/5] Извлекаю аудио…")
        audio_path = audio_mod.extract_audio(src, workdir)
        chunks = audio_mod.split_if_needed(audio_path, workdir)
        _p(f"  аудио: {os.path.getsize(audio_path)/MB:.2f} МБ, кусков={len(chunks)}")

        # 3. Транскрипция
        _p("\n[3/5] Транскрипция…")
        tr = None
        transcription_mode = "mock"
        if groq_key:
            try:
                tr = transcribe.transcribe(chunks, groq_key)
                if tr.words:
                    transcription_mode = "groq"
                    _p(f"  [Groq] язык={tr.language}, слов={len(tr.words)}")
                    preview = tr.text[:120].replace("\n", " ")
                    _p(f"  текст: {preview}…")
                else:
                    tr = None
            except transcribe.TranscribeError as exc:
                _p(f"  [warn] Groq транскрипция не удалась: {exc}")
                tr = None
        if tr is None:
            tr = mock_transcript(s["duration"])
            _p(f"  [MOCK] встроенный транскрипт: слов={len(tr.words)} "
               f"(речь в источнике отсутствует или Groq недоступен)")

        # 4. Выбор моментов
        _p("\n[4/5] Выбор моментов…")
        selection_mode = "naive"
        clips: list[dict] = []
        if groq_key and transcription_mode == "groq":
            try:
                clips = selector.select_clips(tr, s["duration"], groq_key, count=3)
                selection_mode = "groq"
                _p(f"  [Groq LLM] выбрано фрагментов: {len(clips)}")
            except selector.SelectError as exc:
                _p(f"  [warn] Groq выбор не удался: {exc}")
        if not clips:
            clips = naive_clips(s["duration"], count=2)
            _p(f"  [NAIVE] выбрано окон: {len(clips)}")
        for c in clips:
            _p(f"    {c['start']:.1f}–{c['end']:.1f} c ({c['end']-c['start']:.1f} c): "
               f"{c['reason']}")

        # 5. Рендер
        _p("\n[5/5] Рендер клипов…")
        rendered = 0
        for i, clip in enumerate(clips, start=1):
            try:
                summ = render.render_clip(src, clip, tr.words, workdir, i)
            except Exception as exc:  # noqa: BLE001
                ok_all = False
                _p(f"  [FAIL] клип {i}: {exc}")
                continue
            rendered += 1
            line = (
                f"  клип {i}: {summ['width']}x{summ['height']}, "
                f"{summ['duration']:.1f} c, {summ['size_bytes']/MB:.2f} МБ, "
                f"звук={'да' if summ['has_audio'] else 'нет'}, "
                f"субтитров={summ['subs_lines']}"
                + ("  [документом]" if summ['as_document'] else "")
            )
            _p(line)
            report.append(line)
            # Жёсткие проверки корректности клипа.
            if (summ["width"], summ["height"]) != (1080, 1920):
                ok_all = False
                _p(f"    [FAIL] ожидалось 1080x1920, получено "
                   f"{summ['width']}x{summ['height']}")
            if not summ["has_audio"]:
                ok_all = False
                _p("    [FAIL] в клипе нет звука")
            if summ["subs_lines"] < 1:
                ok_all = False
                _p("    [FAIL] в клипе нет субтитров")

        if rendered == 0:
            ok_all = False

        # Итоговый отчёт
        _p("\n" + "=" * 64)
        _p("ОТЧЁТ")
        _p("=" * 64)
        _p(f"Источник:        {'речь (say)' if speech_source else ('TEST_VIDEO' if env_video else 'синтетика')}")
        _p(f"Транскрипция:    {transcription_mode.upper()}"
           + (f" (язык: {tr.language})" if transcription_mode == 'groq' else ""))
        _p(f"Выбор моментов:  {selection_mode.upper()}")
        _p(f"Отрендерено:     {rendered} из {len(clips)}")
        for r in report:
            _p(r.strip())
        _p("-" * 64)
        verdict = "PASS ✅" if ok_all and rendered > 0 else "FAIL ❌"
        _p(f"ИТОГ: {verdict}")
        return 0 if (ok_all and rendered > 0) else 1
    except (FFmpegError, Exception) as exc:  # noqa: BLE001
        _p("\n[FAIL] исключение в пайплайне:")
        _p(textwrap.indent(str(exc), "    "))
        return 1
    finally:
        if keep:
            _p(f"\n[info] рабочая папка сохранена: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
