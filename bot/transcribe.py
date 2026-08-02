"""Транскрипция через Groq (whisper-large-v3-turbo).

- response_format=verbose_json с word-таймкодами;
- поле language берём как есть (ничего не переводим);
- если аудио резалось на куски — склеиваем слова со сдвигом таймкодов.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import httpx

log = logging.getLogger("clipping.transcribe")

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3-turbo"


class TranscribeError(RuntimeError):
    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Transcript:
    language: str
    words: list[Word] = field(default_factory=list)
    text: str = ""


def _humanize(status: int, body: str) -> str:
    if status in (401, 403):
        return "Groq отклонил ключ (401/403). Проверь GROQ_API_KEY."
    if status == 429:
        return "Groq превысил лимит запросов (429). Подожди минуту и попробуй снова."
    if status >= 500:
        return "Groq сейчас недоступен (5xx). Попробуй позже."
    return f"Groq вернул ошибку {status}. Детали в логах."


def _transcribe_one(path: str, api_key: str) -> Transcript:
    """Транскрибировать один аудиофайл. Таймкоды — относительно начала файла."""
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["word", "segment"],
        "temperature": "0",
    }
    try:
        with open(path, "rb") as fh:
            files = {"file": (os.path.basename(path), fh, "audio/mpeg")}
            with httpx.Client(timeout=600) as client:
                resp = client.post(GROQ_URL, headers=headers, data=data, files=files)
    except httpx.HTTPError as exc:
        log.exception("Сетевая ошибка при обращении к Groq")
        raise TranscribeError(
            "Не удалось связаться с Groq (сеть). Проверь интернет на хостинге."
        ) from exc

    if resp.status_code != 200:
        tail = resp.text[-1500:]
        log.error("Groq transcription %d: %s", resp.status_code, tail)
        raise TranscribeError(_humanize(resp.status_code, tail))

    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise TranscribeError("Groq вернул некорректный JSON транскрипции.") from exc

    language = (payload.get("language") or "").strip() or "unknown"
    words: list[Word] = []
    for w in payload.get("words", []) or []:
        try:
            words.append(Word(
                word=str(w.get("word", "")).strip(),
                start=float(w["start"]),
                end=float(w["end"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    text = (payload.get("text") or "").strip()
    return Transcript(language=language, words=words, text=text)


def transcribe(chunks: list[tuple[str, float]], api_key: str) -> Transcript:
    """Транскрибировать один или несколько кусков, склеить со сдвигом таймкодов.

    chunks — список (путь, offset_секунд) из audio.split_if_needed().
    """
    if not chunks:
        raise TranscribeError("Нет аудио для транскрипции.")

    merged = Transcript(language="unknown", words=[], text="")
    texts: list[str] = []
    for i, (path, offset) in enumerate(chunks):
        log.info("Транскрибирую кусок %d/%d (offset=%.1f c): %s",
                 i + 1, len(chunks), offset, path)
        part = _transcribe_one(path, api_key)
        if i == 0:
            merged.language = part.language
        elif merged.language == "unknown" and part.language != "unknown":
            merged.language = part.language
        for w in part.words:
            merged.words.append(Word(w.word, w.start + offset, w.end + offset))
        if part.text:
            texts.append(part.text)

    merged.text = " ".join(texts).strip()
    if not merged.words:
        raise TranscribeError(
            "Транскрипт пустой — в видео не распозналась речь. "
            "Возможно, там только музыка или тишина."
        )
    log.info("Транскрипт готов: язык=%s, слов=%d", merged.language, len(merged.words))
    return merged
