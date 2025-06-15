# PokerRAG Telegram Bot

**PokerRAG** is a lightweight, production-ready Telegram bot that provides instant answers to poker-related questions using semantic search powered by Sentence Transformers.

---

## 🚀 Features

- 💬 **Natural language QA**: Users can ask poker-related questions in free text.
- 🧠 **Semantic search** with [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)
- 📉 **Smart filtering**: Only answers when confidence score > 0.78
- 🤖 **Telegram bot interface** with [`python-telegram-bot==20.6`](https://docs.python-telegram-bot.org/en/stable/)
- 📚 **Pluggable JSON knowledge base** (easy to expand)
- 🧪 **Shows top-3 related questions** even for rejected queries

---

## 📦 Tech Stack

| Component        | Library / Tool                          |
| ---------------- | --------------------------------------- |
| Embeddings       | `sentence-transformers` (`e5-base`)     |
| Bot framework    | `python-telegram-bot` (v20.6)           |
| Telegram API     | `httpx` (indirectly used)               |
| QA retrieval     | Cosine similarity over embedded vectors |
| Language support | Multilingual                            |

---

## 🛠 Installation

```bash
git clone https://github.com/Vlad-Boyar/PokerRAG.git
cd PokerRAG
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Add your Telegram Bot Token to a `.env` file:

```
TELEGRAM_TOKEN=your_token_here
```

---

## ▶️ Run the bot

```bash
python main.py
```

---

## ✍️ Example usage

**User**: `Who is the founder of your school?`

**Bot**:

```
Daniel Clark, a professional poker player, founded our school.

🔍 Related questions:
1. Who founded your poker school? — 📊 0.83
2. Who are the main coaches? — 📊 0.81
3. What is your mission? — 📊 0.78
```

**User**: `pisya popa`

**Bot**:

```
Sorry, I couldn't understand your question. Please try rephrasing.

🔍 Related questions:
1. What do you teach? — 📊 0.62
2. How do I sign up? — 📊 0.60
3. Is there a leaderboard? — 📊 0.59
```

---

## 📁 Project structure

```
PokerRAG/
├── main.py              # Telegram bot main logic
├── qa_utils.py          # Embedding + similarity scoring
├── questions.json       # Knowledge base
├── requirements.txt
├── .env                 # Telegram token (not in repo)
└── README.md
```

---

## 🤝 Credits

Created by **Vlad Boyar** — poker player, automation engineer, and AI enthusiast.

Project goals:

- Build a real-world AI project
- Learn fast embedding search
- Add to GitHub + LinkedIn portfolio

---

## 🌐 Links

- [LinkedIn profile](https://www.linkedin.com/in/Vlad-Boyarin)
- [GitHub repository](https://github.com/Vlad-Boyar/PokerRAG)
- [HuggingFace model](https://huggingface.co/intfloat/multilingual-e5-base)
