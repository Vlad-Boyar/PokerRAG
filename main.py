import os
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import Document

import asyncio

# Load .env
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Настройки LlamaIndex
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

# Подготовка индекса
def build_index_from_csv(csv_path="faq.csv"):
    df = pd.read_csv(csv_path)
    documents = [Document(text=f"Q: {row['question']}\nA: {row['answer']}") for _, row in df.iterrows()]
    return VectorStoreIndex.from_documents(documents)

index = build_index_from_csv()
query_engine = index.as_query_engine()

# Ответ пользователю
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    print(f"[RAG] Вопрос от пользователя: {question}")
    response = query_engine.query(question)
    await update.message.reply_text(str(response))

# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот-помощник. Спроси что-нибудь — и я найду ответ!")

# Главный async run
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    print("🤖 Бот запущен и подключён к базе знаний.")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
