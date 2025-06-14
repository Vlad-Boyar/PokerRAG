import csv
import os
from datetime import datetime

LOG_FILE = "logs.csv"

# Если файл ещё не существует — создаём с заголовками
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "question", "answer", "source", "score"])

def log_interaction(question: str, answer: str, source: str = "RAG", score: float = 1.0):
    timestamp = datetime.now().isoformat()
    row = [timestamp, question, answer, source, score]

    with open(LOG_FILE, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
