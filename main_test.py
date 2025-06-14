import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Бот работает!")

async def main():
    print("🟢 Запуск Telegram бота...")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))

    await app.initialize()
    await app.start()
    print("🤖 Бот запущен и слушает /start")

    # Бесконечно ждем, пока бот работает
    await asyncio.Event().wait()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # fallback для уже запущенного event loop (например, в VS Code)
        print("⚠️ Event loop уже активен, fallback...")
        loop = asyncio.get_event_loop()
        loop.create_task(main())
