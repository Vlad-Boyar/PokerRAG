import os, pandas as pd, asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

def load_documents_from_csv(path="faq.csv"):
    df = pd.read_csv(path)
    return [Document(text=f"Q: {r.question}\nA: {r.answer}") for _, r in df.iterrows()]

documents = load_documents_from_csv()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=3)

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    nodes = retriever.retrieve(query)
    context_str = "\n\n".join(node.get_content() for node in nodes)
    prompt = f"Контекст:\n{context_str}\n\nВопрос: {query}\nОтвет:"
    resp = Settings.llm.complete(prompt).text.strip()
    await update.message.reply_text(resp)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Спроси что-нибудь.")

async def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    await app.initialize(); await app.start(); await app.updater.start_polling()
    print("Бот запущен.")
    while True: await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
