import os
import logging
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import langdetect

# ========== Настройка ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ========== Классы ==========
class Query(BaseModel):
    query: str

# ========== Загрузка данных ==========
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME))
retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=5)
reranker = SentenceTransformerRerank(model=SentenceTransformer(RERANK_MODEL_NAME), top_n=3)

# ========== Перевод ==========
def detect_lang(text):
    try:
        return langdetect.detect(text)
    except:
        return "en"

def translate(text, source, target):
    if source == target:
        return text
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except:
        return text

# ========== Ответ ==========
def answer_query(user_query: str):
    user_lang = detect_lang(user_query)
    logger.info(f"[User] {user_query} ({user_lang})")

    query_en = translate(user_query, user_lang, "en")
    nodes = retriever.retrieve(query_en)
    reranked = reranker.postprocess_nodes(nodes, query_str=query_en)

    top_node = reranked[0] if reranked else None
    top_score = top_node.score if top_node else 0
    logger.info(f"Top score: {top_score:.4f}")

    if top_node and top_score > 0.78:
        q, a = top_node.node.metadata.get("question"), top_node.node.metadata.get("answer")
        translated_answer = translate(f"Q: {q}\nA: {a}", source="en", target=user_lang)
        return translated_answer
    
    # иначе просто похожие вопросы
    top_questions = [n.node.metadata.get("question") for n in reranked]
    translated_qs = [translate(q, source="en", target=user_lang) for q in top_questions]
    alt_suggestions = "\n".join(f"\u2022 {q}" for q in translated_qs)
    fallback_msg = translate("❌ Nothing found. Maybe you meant:", source="en", target=user_lang)
    return f"{fallback_msg}\n{alt_suggestions}"

# ========== API ==========
app = FastAPI()

@app.post("/query")
async def query_route(req: Query):
    reply = answer_query(req.query)
    return {"answer": reply}

# ========== Run ==========
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)