"""Очередь задач: asyncio.Queue + РОВНО один воркер.

- одновременно обрабатывается только одна задача;
- не больше MAX_PER_USER задач в очереди от одного пользователя;
- при постановке возвращаем позицию в очереди.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Awaitable, Callable

log = logging.getLogger("clipping.queue")

MAX_PER_USER = 3


@dataclass
class Job:
    user_id: int
    chat_id: int
    url: str
    status_message_id: int | None = None


Processor = Callable[[Job], Awaitable[None]]


class TaskQueue:
    def __init__(self, processor: Processor):
        self._q: asyncio.Queue[Job] = asyncio.Queue()
        self._processor = processor
        self._counts: dict[int, int] = defaultdict(int)
        self._worker: asyncio.Task | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="clip-worker")
            log.info("Воркер очереди запущен.")

    def user_count(self, user_id: int) -> int:
        return self._counts[user_id]

    def try_enqueue(self, job: Job) -> int:
        """Поставить задачу. Вернуть позицию (>=1) или -1, если лимит исчерпан."""
        if self._counts[job.user_id] >= MAX_PER_USER:
            return -1
        self._counts[job.user_id] += 1
        self._q.put_nowait(job)
        # Позиция = сколько задач сейчас ждут (включая эту).
        return self._q.qsize()

    async def _run(self) -> None:
        while True:
            job = await self._q.get()
            try:
                await self._processor(job)
            except Exception:  # noqa: BLE001 — воркер не должен умирать
                log.exception("Необработанная ошибка в воркере для job=%s", job)
            finally:
                self._counts[job.user_id] = max(0, self._counts[job.user_id] - 1)
                self._q.task_done()
