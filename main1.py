import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    filters,
)
import asyncio

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Ответ пользователю
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[DEBUG] Получено сообщение: {update.message.text}")
    await update.message.reply_text("✅ Бот живой и отвечает!")

# /start команда
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот, пиши мне :)")

async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    print("🤖 Бот запущен. Ожидаю сообщений в Telegram...")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    # оставим бесконечный цикл вручную (вместо run_forever())
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
