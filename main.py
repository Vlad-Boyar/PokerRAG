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

RELEVANCE_THRESHOLD = 0.78

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
    if not nodes:
        await update.message.reply_text("❌ Sorry, I couldn't find an answer.\nPlease try rephrasing your question.")
        return

    # Подбор релевантных узлов
    best_node = nodes[0]
    is_relevant = best_node.score and best_node.score >= RELEVANCE_THRESHOLD

    # Формируем блок похожих вопросов
    similar = "\n\n🔍 Related questions:\n"
    for i, node in enumerate(nodes[:3]):
        q = node.get_content().split("\n")[0].replace("Q: ", "").strip()
        score = node.score if node.score else 0
        similar += f"{i+1}. {q} — 📊 {score:.2f}\n"

    # Формируем финальный ответ
    if is_relevant:
        context_str = best_node.get_content()
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        response = Settings.llm.complete(prompt).text.strip()
        final_reply = f"{response}{similar}"
    else:
        final_reply = f"🤷 Sorry, I couldn't find a confident answer.\nPlease try rephrasing your question.{similar}"

    await update.message.reply_text(final_reply)

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
