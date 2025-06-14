import os
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from llama_index.core import Settings, Document
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index.llms.openai import OpenAI

import asyncio

# Load .env
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Настройки LLM
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

# Загрузка базы
def load_documents_from_csv(csv_path="faq.csv"):
    df = pd.read_csv(csv_path)
    return [Document(text=f"Q: {row['question']}\nA: {row['answer']}") for _, row in df.iterrows()]

documents = load_documents_from_csv()

# BM25 Retriever
retriever = BM25Retriever.from_defaults(documents)

# Реранкер на базе BAAI/bge-reranker-base
reranker = SentenceTransformerRerank(top_n=1, model="BAAI/bge-reranker-base")

# Ответ пользователю
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    print(f"[RAG] Вопрос от пользователя: {question}")

    nodes = retriever.retrieve(question)
    reranked_nodes = reranker.postprocess_nodes(nodes, query=question)

    answer = reranked_nodes[0].get_content() if reranked_nodes else "Не нашёл подходящего ответа."
    await update.message.reply_text(answer)

# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот-помощник. Спроси что-нибудь — и я найду ответ!")

# Главный run
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    print("🤖 Бот запущен и готов к ответам.")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
