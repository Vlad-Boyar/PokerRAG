from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import pandas as pd
import os

# Загрузка .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY не найден!")

# Загрузка CSV
df = pd.read_csv("faq.csv")

# Преобразование в документы
documents = [
    Document(text=row["answer"], metadata={"question": row["question"]})
    for _, row in df.iterrows()
]

# Установка глобальных настроек
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding()

# Построение индекса
index = VectorStoreIndex.from_documents(documents)

# Запуск query engine
query_engine = index.as_query_engine()

# Интерфейс
print("🤖 Введи вопрос (или 'exit' для выхода):")
while True:
    user_input = input("❓ ")
    if user_input.lower() in ("exit", "quit"):
        break
    response = query_engine.query(user_input)
    print("💬", response)
