"""Точка входа: aiogram v3, long polling.

При старте: проверяем ffmpeg и переменные окружения (иначе падаем с внятным
сообщением в лог). Дальше — приём ссылок, очередь с одним воркером, отправка
клипов по мере готовности.
"""
from __future__ import annotations

import asyncio
import logging
import re

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message

from . import config, pipeline
from .task_queue import MAX_PER_USER, Job, TaskQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("clipping.main")

YT_RE = re.compile(
    r"(https?://)?(www\.|m\.)?(youtube\.com/\S+|youtu\.be/\S+)", re.IGNORECASE
)

WELCOME = (
    "👋 Привет! Пришли мне ссылку на видео с YouTube, и я нарежу из него "
    "вертикальные клипы 9:16 с субтитрами.\n\n"
    "• Длина исходника — до 90 минут.\n"
    "• В очереди одновременно до {limit} твоих задач.\n"
    "• Обрабатываю по одной задаче за раз, наберись терпения 🙂"
).format(limit=MAX_PER_USER)

MB = 1024 * 1024


def _clip_caption(summary: dict) -> str:
    w, h = summary.get("width"), summary.get("height")
    dur = summary.get("duration") or 0
    size_mb = (summary.get("size_bytes") or 0) / MB
    reason = (summary.get("reason") or "").strip()
    head = "🎬 Клип" + ("  (документом — большой размер)" if summary.get("as_document") else "")
    specs = f"⏱ {dur:.0f} c · 🖼 {w}×{h} · 💾 {size_mb:.1f} МБ"
    body = f"\n\n💡 {reason}" if reason else ""
    return f"{head}\n{specs}{body}"


async def _make_processor(bot: Bot, cfg: config.Config):
    async def process_job(job: Job) -> None:
        async def on_status(text: str) -> None:
            try:
                await bot.send_message(job.chat_id, text)
            except Exception:  # noqa: BLE001
                log.warning("Не смог отправить статус в чат %s", job.chat_id)

        async def on_clip(summary: dict) -> None:
            file = FSInputFile(summary["path"])
            caption = _clip_caption(summary)
            try:
                if summary.get("as_document"):
                    await bot.send_document(job.chat_id, file, caption=caption)
                else:
                    await bot.send_video(
                        job.chat_id, file, caption=caption,
                        width=summary.get("width"), height=summary.get("height"),
                        duration=int(summary.get("duration") or 0),
                        supports_streaming=True,
                    )
            except Exception:  # noqa: BLE001
                log.exception("Не удалось отправить клип в чат %s", job.chat_id)
                await on_status("⚠️ Клип собрался, но не отправился в Telegram.")

        try:
            n = await pipeline.process(job.url, cfg, on_status, on_clip)
            await on_status(f"✅ Готово! Отправлено клипов: {n}.")
        except pipeline.PipelineError as exc:
            await on_status(f"❌ {exc.user_message}")
        except Exception as exc:  # noqa: BLE001
            log.exception("Непредвиденная ошибка обработки job")
            await on_status(
                "❌ Непредвиденная ошибка при обработке. Попробуй другое видео "
                "или повтори позже. Детали — в логах бота."
            )

    return process_job


def build_dispatcher(cfg: config.Config, queue: TaskQueue) -> Dispatcher:
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def on_start(message: Message) -> None:
        await message.answer(WELCOME)

    @dp.message(Command("help"))
    async def on_help(message: Message) -> None:
        await message.answer(WELCOME)

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        text = message.text or ""
        match = YT_RE.search(text)
        if not match:
            await message.answer("Пришли ссылку на видео с YouTube 🙂")
            return
        url = match.group(0)
        if not url.lower().startswith("http"):
            url = "https://" + url

        user_id = message.from_user.id if message.from_user else 0
        if queue.user_count(user_id) >= MAX_PER_USER:
            await message.answer(
                f"У тебя уже {MAX_PER_USER} задач(и) в работе — дождись их завершения 🙏"
            )
            return

        job = Job(user_id=user_id, chat_id=message.chat.id, url=url)
        pos = queue.try_enqueue(job)
        if pos < 0:
            await message.answer(
                f"У тебя уже {MAX_PER_USER} задач(и) в работе — дождись их завершения 🙏"
            )
            return
        if pos <= 1:
            await message.answer("✅ Принял, в очереди. Начинаю обработку…")
        else:
            await message.answer(f"✅ Принял, в очереди. Твоя позиция: {pos}.")

    return dp


async def main() -> None:
    config.check_ffmpeg()
    cfg = config.load_config()

    bot = Bot(cfg.bot_token)
    processor = await _make_processor(bot, cfg)
    queue = TaskQueue(processor)
    queue.start()

    dp = build_dispatcher(cfg, queue)

    if cfg.admin_id:
        try:
            await bot.send_message(cfg.admin_id, "🤖 Бот запущен и готов к работе.")
        except Exception:  # noqa: BLE001
            log.warning("Не смог написать админу %s", cfg.admin_id)

    log.info("Старт polling.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Остановлено.")
