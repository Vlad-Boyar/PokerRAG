import pandas as pd
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# Загрузка ключей
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not OPENAI_KEY or not TELEGRAM_TOKEN:
    raise ValueError("❌ Не хватает ключей в .env!")

# Настройка LlamaIndex
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding()

# Загрузка базы из CSV
df = pd.read_csv("faq.csv")
documents = [
    Document(text=row["answer"], metadata={"question": row["question"]})
    for _, row in df.iterrows()
]
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Обработчик сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        print(f"[DEBUG] Пришло сообщение: {update.message.text}")
        await update.message.reply_text("Ответ получен")
    else:
        print("[DEBUG] Пришло что-то, но не текстовое сообщение")

# --- ЗАПУСК НА WINDOWS ---
import asyncio

app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT, handle_message))

print("🤖 Бот запущен. Ожидает сообщения в Telegram...")
curl https://api.telegram.org/bot<your_token>/getMe
if os.name == "nt":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(app.initialize())
    loop.create_task(app.start())
    loop.run_forever()
else:
    asyncio.run(app.run_polling())
