"""Выбор ярких фрагментов через Groq LLM (llama-3.3-70b-versatile).

Отдаём модели транскрипт с таймкодами, получаем строгий JSON, валидируем:
границы внутри длительности, end>start, длина 30..60 c (с небольшим допуском).
"""
from __future__ import annotations

import json
import logging

import httpx

from .transcribe import Transcript, Word

log = logging.getLogger("clipping.selector")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

MIN_LEN = 30.0
MAX_LEN = 60.0
# Небольшой допуск: модель редко попадает секунда-в-секунду.
LEN_TOLERANCE = 8.0


class SelectError(RuntimeError):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


def _build_timestamped_text(words: list[Word], line_gap: float = 6.0) -> str:
    """Собрать компактный транскрипт: строки вида [start-end] текст."""
    if not words:
        return ""
    lines: list[str] = []
    buf: list[str] = []
    line_start = words[0].start
    line_end = words[0].end
    for w in words:
        if buf and (w.end - line_start) >= line_gap:
            lines.append(f"[{line_start:.1f}-{line_end:.1f}] {' '.join(buf)}")
            buf = []
            line_start = w.start
        buf.append(w.word)
        line_end = w.end
    if buf:
        lines.append(f"[{line_start:.1f}-{line_end:.1f}] {' '.join(buf)}")
    return "\n".join(lines)


def _clamp_to_words(start: float, end: float, words: list[Word]) -> tuple[float, float]:
    """Подвинуть границы к ближайшим словам, чтобы не резать посреди слова."""
    starts = [w.start for w in words]
    ends = [w.end for w in words]
    near_start = min(starts, key=lambda s: abs(s - start))
    near_end = min(ends, key=lambda e: abs(e - end))
    if near_end <= near_start:
        near_end = start + MIN_LEN
    return near_start, near_end


def select_clips(
    transcript: Transcript,
    video_duration: float,
    api_key: str,
    count: int,
) -> list[dict]:
    """Вернуть список {start, end, reason}. Всегда хотя бы один валидный фрагмент."""
    ts_text = _build_timestamped_text(transcript.words)
    if not ts_text:
        raise SelectError("Пустой транскрипт — нечего выбирать.")

    want = max(3, min(5, count if count else 3))
    system = (
        "Ты монтажёр коротких вертикальных видео (Reels/Shorts). "
        "Тебе дают транскрипт с таймкодами в секундах. "
        "Выбери самодостаточные фрагменты по 30–60 секунд: законченная мысль, "
        "сильное цепляющее начало, без обрыва на полуслове. "
        "Отвечай СТРОГО JSON-объектом вида "
        '{\"clips\": [{\"start\": число, \"end\": число, \"reason\": строка}]}. '
        "reason — короткое объяснение на русском, почему фрагмент цепляет. "
        "start и end — в секундах из таймкодов. Никакого текста вне JSON."
    )
    user = (
        f"Длительность видео: {video_duration:.1f} c. "
        f"Нужно {want} фрагментов по 30–60 секунд.\n\n"
        f"Транскрипт:\n{ts_text}"
    )

    body = {
        "model": MODEL,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=180) as client:
            resp = client.post(GROQ_URL, headers=headers, json=body)
    except httpx.HTTPError as exc:
        log.exception("Сетевая ошибка при обращении к Groq LLM")
        raise SelectError("Не удалось связаться с Groq (LLM). Попробуй позже.") from exc

    if resp.status_code != 200:
        tail = resp.text[-1500:]
        log.error("Groq chat %d: %s", resp.status_code, tail)
        if resp.status_code in (401, 403):
            raise SelectError("Groq отклонил ключ на этапе выбора моментов (401/403).")
        if resp.status_code == 429:
            raise SelectError("Groq лимит (429) на этапе выбора. Подожди минуту.")
        raise SelectError(f"Groq вернул ошибку {resp.status_code} на выборе моментов.")

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        log.error("Не разобрать ответ LLM: %s", resp.text[:800])
        raise SelectError("Groq вернул некорректный JSON на выборе моментов.") from exc

    raw = parsed.get("clips") if isinstance(parsed, dict) else parsed
    if not isinstance(raw, list):
        raise SelectError("LLM не вернул список фрагментов.")

    valid = _validate(raw, transcript.words, video_duration)
    if not valid:
        # Фолбэк: берём фрагмент от сильного места (начало речи) на MIN_LEN.
        log.warning("LLM не дал валидных фрагментов — использую фолбэк.")
        valid = _fallback(transcript.words, video_duration)
        if not valid:
            raise SelectError("Не удалось выбрать ни одного корректного фрагмента.")

    valid = valid[:want]
    log.info("Выбрано фрагментов: %d", len(valid))
    return valid


def _validate(raw: list, words: list[Word], video_duration: float) -> list[dict]:
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        reason = str(item.get("reason", "")).strip() or "Яркий момент."

        if end <= start:
            continue
        # Обрезаем к границам видео.
        start = max(0.0, start)
        end = min(video_duration, end) if video_duration > 0 else end
        if end <= start:
            continue

        length = end - start
        if length < MIN_LEN - LEN_TOLERANCE or length > MAX_LEN + LEN_TOLERANCE:
            continue

        # Подгоняем к словам, чтобы не рвать речь.
        if words:
            start, end = _clamp_to_words(start, end, words)
            if video_duration > 0:
                end = min(end, video_duration)
            if end - start < MIN_LEN - LEN_TOLERANCE:
                continue

        out.append({"start": round(start, 2), "end": round(end, 2), "reason": reason})

    # Убираем сильные пересечения (оставляем первый из пары).
    out.sort(key=lambda c: c["start"])
    deduped: list[dict] = []
    for c in out:
        if deduped and c["start"] < deduped[-1]["end"] - 5:
            continue
        deduped.append(c)
    return deduped


def _fallback(words: list[Word], video_duration: float) -> list[dict]:
    if not words:
        return []
    start = words[0].start
    end = min(start + 45.0, video_duration if video_duration > 0 else start + 45.0)
    if end - start < MIN_LEN - LEN_TOLERANCE:
        end = start + MIN_LEN
    return [{"start": round(start, 2), "end": round(end, 2),
             "reason": "Начало ролика (автоматический выбор)."}]
