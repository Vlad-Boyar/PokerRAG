import os
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from llama_index.core import Document, Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.sentence_transformer_rerank.sentence_transformer_rerank import SentenceTransformerRerank
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

import asyncio

# Load .env
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Настройки LLM + эмбеддер
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

# Загрузка базы
def load_documents_from_csv(csv_path="faq.csv"):
    df = pd.read_csv(csv_path)
    return [Document(text=f"Q: {row['question']}\nA: {row['answer']}") for _, row in df.iterrows()]

documents = load_documents_from_csv()

# BM25 + reranker
retriever = BM25Retriever.from_defaults(documents)
reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=1)

# Обработка запроса
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    print(f"[RAG] Вопрос от пользователя: {question}")

    nodes = retriever.retrieve(question)
    reranked_nodes = reranker.postprocess_nodes(nodes, query=question)

    if reranked_nodes:
        response = reranked_nodes[0].get_content()
    else:
        response = "Извините, я не нашёл подходящего ответа."

    await update.message.reply_text(response)

# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот-помощник. Спроси что-нибудь — и я найду ответ!")

# Запуск Telegram‑бота
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
