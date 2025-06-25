import os
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from deep_translator import GoogleTranslator
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# === SETUP ===
load_dotenv()
logging.basicConfig(level=logging.INFO)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

top_languages = {'en', 'es', 'fr', 'de', 'pt', 'ru', 'zh', 'ar', 'hi', 'ja'}

model_embed = SentenceTransformer("intfloat/multilingual-e5-base")
model_rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

df = pd.read_csv("faq.csv")
questions = df["question"].tolist()
answers = df["answer"].tolist()
question_embeddings = model_embed.encode(
    [f"query: {q}" for q in questions], convert_to_tensor=True, show_progress_bar=False
)

# === LANGUAGE TOOLS ===
def detect_language(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text, return_all=True)["src"]
    except Exception:
        return "en"

def translate(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text
    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except:
        return text

# === BOT LOGIC ===
def get_response(user_query: str) -> str:
    detected_lang = detect_language(user_query)
    query_embed = model_embed.encode(f"query: {user_query}", convert_to_tensor=True)
    scores = np.dot(question_embeddings, query_embed)
    top_indices = np.argsort(-scores)[:3]

    candidate_qas = [(questions[i], answers[i], float(scores[i])) for i in top_indices]
    pairs = [[user_query, q] for q, _, _ in candidate_qas]
    rerank_scores = model_rerank.predict(pairs)

    reranked = sorted(zip(candidate_qas, rerank_scores), key=lambda x: -x[1])
    (best_q, best_a, _), best_score = reranked[0]

    if best_score > 0.78:
        prompt = f"""You are a helpful assistant that answers ONLY using the provided context.
If the context does not contain the answer, say "I don't know".

Context:
Q: {best_q}
A: {best_a}

User Question: {user_query}
Answer:"""
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content.strip()
        translated = translate(response, "en", detected_lang) if detected_lang in top_languages and detected_lang != "en" else response
        return translated
    else:
        suggestions = [q for (q, _, _), _ in reranked]
        translated = [translate(q, "en", detected_lang) if detected_lang in top_languages and detected_lang != "en" else q for q in suggestions]
        return "❌ Nothing found. You might be looking for:\n" + "\n".join(f"• {q}" for q in translated)

# === TELEGRAM HANDLER ===
async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message.text
    logging.info(f"[User] {user_msg}")
    reply = get_response(user_msg)
    await update.message.reply_text(reply)

# === START BOT ===
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    print("✅ Bot is running...")
    app.run_polling()
