import os
import logging
import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import asyncio

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# LLM and embeddings
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_KEY)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-base",
    query_instruction="query:",
    text_instruction="passage:"
)

# Load FAQ
def load_faq(csv_path="faq.csv"):
    df = pd.read_csv(csv_path)
    return [Document(text=f"Q: {row['question']}\nA: {row['answer']}") for _, row in df.iterrows()]

documents = load_faq()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=5)

# Handlers
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    logger.info(f"User: {query}")
    nodes = retriever.retrieve(query)

    if not nodes or nodes[0].score < 0.5:
        await update.message.reply_text("❌ I couldn't find a confident answer.")
        return

    context_str = nodes[0].get_content()
    prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
    response = Settings.llm.complete(prompt).text.strip()

    similar = "\n\n🔍 Related questions:\n"
    for i, node in enumerate(nodes[:3]):
        q = node.get_content().split("\n")[0].replace("Q: ", "").strip()
        similar += f"{i+1}. {q} — 📊 {node.score:.2f}\n"

    await update.message.reply_text(f"{response}{similar}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hi! Ask me anything about poker coaching.")

# Main startup
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    logger.info("🤖 Bot is running and ready.")
    app.run_polling()  # <--- No asyncio.run() here!

if __name__ == "__main__":
    main()
