import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Загружаем модель
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Загружаем CSV
df = pd.read_csv("faq.csv")

# Твой запрос
query_text = "пися попа"
query_embed = model.encode("query: " + query_text, convert_to_tensor=True)

print(f"Q: {query_text}\n")

# Перебираем базу
for _, row in df.iterrows():
    text = f"passage: Q: {row['question']}\nA: {row['answer']}"
    doc_embed = model.encode(text, convert_to_tensor=True)

    score = util.cos_sim(query_embed, doc_embed).item()

    print(f"A: {row['answer']}")
    print(f"score = {score:.4f}")
    print("-" * 40)
