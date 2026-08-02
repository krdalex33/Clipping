"""Генерация .ass субтитров из word-таймкодов.

- по 3–5 слов на строку;
- крупный жирный шрифт, белый текст, чёрная обводка;
- позиция снизу с отступом ~15% высоты кадра (1920 -> ~288 px).

Таймкоды в .ass — относительно НАЧАЛА клипа (клип вырезан с -ss, стартует с 0)
и ДО ускорения. Ускорение (setpts) применяется после ass, поэтому субтитры
уезжают вместе со своими кадрами и остаются синхронными.
"""
from __future__ import annotations

import logging

from .transcribe import Word

log = logging.getLogger("clipping.subtitles")

FONT_NAME = "Montserrat"
FONT_SIZE = 76
OUTLINE = 4
WORDS_PER_LINE_MIN = 3
WORDS_PER_LINE_MAX = 4
# Максимальная пауза между словами внутри одной строки (сек). Больше — новая строка.
MAX_GAP = 0.8

PLAY_W = 1080
PLAY_H = 1920
MARGIN_V = int(PLAY_H * 0.15)  # ~288 px снизу


def _fmt_time(t: float) -> str:
    """Секунды -> H:MM:SS.cc (сотые доли), формат ASS."""
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    # В диалоге ASS фигурные скобки — служебные, экранируем.
    return text.replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def _group_words(words: list[Word]) -> list[list[Word]]:
    """Разбить слова на строки по 3–5 с учётом пауз."""
    lines: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if not w.word:
            continue
        if cur:
            gap = w.start - cur[-1].end
            if len(cur) >= WORDS_PER_LINE_MAX or (len(cur) >= WORDS_PER_LINE_MIN and gap > MAX_GAP):
                lines.append(cur)
                cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def build_ass(words: list[Word], clip_start: float, clip_end: float, out_path: str) -> int:
    """Собрать .ass для окна [clip_start, clip_end]. Вернуть число строк субтитров."""
    clip_len = max(0.0, clip_end - clip_start)
    # Слова, попадающие в окно клипа; сдвигаем к нулю.
    local: list[Word] = []
    for w in words:
        if w.end <= clip_start or w.start >= clip_end:
            continue
        s = max(0.0, w.start - clip_start)
        e = min(clip_len, w.end - clip_start)
        if e <= s:
            e = min(clip_len, s + 0.3)
        local.append(Word(word=w.word, start=s, end=e))

    lines = _group_words(local)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_W}
PlayResY: {PLAY_H}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{FONT_NAME},{FONT_SIZE},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{OUTLINE},0,2,60,60,{MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for line in lines:
        if not line:
            continue
        start = line[0].start
        end = line[-1].end
        if end <= start:
            end = start + 0.4
        text = _escape(" ".join(w.word for w in line))
        if not text:
            continue
        events.append(
            f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Sub,,0,0,0,,{text}"
        )

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write("\n".join(events))
        fh.write("\n")

    log.info("Субтитры: %s (%d строк)", out_path, len(events))
    return len(events)
