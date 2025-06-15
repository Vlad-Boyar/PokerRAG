import os
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, ContextTypes,
    CommandHandler, MessageHandler, filters
)
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Загрузка токенов
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Настройка LLM и эмбеддера
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

# Обработка сообщений
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    print(f"[RAG] Вопрос от пользователя: {query}")
    nodes = retriever.retrieve(query)

    if not nodes or (nodes[0].score or 0.0) < 0.5:
        await update.message.reply_text("❌ Не нашёл уверенного ответа.")
        return

    context_str = nodes[0].get_content()
    prompt = f"Контекст:\n{context_str}\n\nВопрос: {query}\nОтвет:"
    response = Settings.llm.complete(prompt).text.strip()

    explanation = "🔍 Похожие вопросы:\n"
    for i, node in enumerate(nodes[:3]):
        question_line = node.get_content().split("\n")[0].replace("Q: ", "").strip()
        explanation += f"{i+1}. {question_line}  —  📊 {node.score:.2f}\n"

    await update.message.reply_text(f"{response}\n\n{explanation}")

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Напиши вопрос, и я постараюсь найти ответ в базе.")

# 👇 Запуск напрямую через run_polling()
if __name__ == "__main__":
    print("🤖 Бот запущен и ждёт вопросов.")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.run_polling()  # <--- sync вызов, всё ок даже если loop уже работает
