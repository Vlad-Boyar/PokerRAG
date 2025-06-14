import os
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import asyncio

# Загрузка токенов
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Настройка LLM и эмбеддеров
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-base",
    embed_batch_size=8,
    query_instruction="query:",
    text_instruction="passage:"
)

# Загрузка базы
def load_documents_from_csv(csv_path="faq.csv"):
    df = pd.read_csv(csv_path)
    return [Document(text=f"Q: {row['question']}\nA: {row['answer']}") for _, row in df.iterrows()]

documents = load_documents_from_csv()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=3)

# Обработка запроса
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    print(f"[RAG] Вопрос от пользователя: {query}")

    nodes = retriever.retrieve(query)
    if not nodes:
        await update.message.reply_text("❌ Не нашёл подходящего ответа.")
        return

    top_node = nodes[0]
    top_score = top_node.score or 0.0

    # Порог галлюцинации
    if top_score < 0.5:
        await update.message.reply_text("❌ Я не уверен в ответе — в базе нет близких совпадений.")
        return

    # Формирование промпта
    context_str = top_node.get_content()
    prompt = f"Контекст:\n{context_str}\n\nВопрос: {query}\nОтвет:"
    response = Settings.llm.complete(prompt).text.strip()

    # Вывод топ-3 вопросов с score
    explanation = "🔍 Похожие вопросы:\n"
    for i, node in enumerate(nodes[:3]):
        score = node.score or 0.0
        question_line = node.get_content().split("\n")[0].replace("Q: ", "").strip()
        explanation += f"{i+1}. {question_line}  —  📊 {score:.2f}\n"

    final_message = f"{response}\n\n{explanation}"
    await update.message.reply_text(final_message)

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Спроси что-нибудь — и я постараюсь найти ответ в базе знаний.")

# Запуск бота
async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    print("🤖 Бот запущен и ждёт вопросов.")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
