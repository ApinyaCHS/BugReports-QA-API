# 🧩 Bug Reports QA API – Intelligent Retrieval and Summarization

This project provides a **FastAPI** interface for intelligent question-answering over internal **bug reports** or **feedback documents**.  
It combines **LangGraph**, **FAISS**, and **Ollama-based LLMs** to analyze, retrieve, and summarize relevant bug information.

---

## ⚙️ System Overview

### 🧠 Architecture
The pipeline uses **two cooperating LLMs** orchestrated through **LangGraph**:

1. **LLM Router (Decision Model)**  
   - Acts as the first reasoning layer.  
   - Decides whether the user’s query:
     - requires **document retrieval** (`query` mode), or  
     - can be **answered directly** (`response` mode).  
   - This is determined based on whether the question references bugs, products, or internal feedback.

2. **LLM Generator (Summarization Model)**  
   - If the first model chooses `query`, it calls a retrieval tool that searches a **FAISS vector database** (semantic index).  
   - The retriever finds the **top-5 most similar documents** to the question using embeddings from `Qwen3-Embedding-8B`.
   - The second LLM (`Qwen3-30B-Instruct`) then reads these retrieved documents and:
     - Summarizes the relevant bug issue,
     - Explains the context or cause,
     - Outputs the final structured result as **JSON**.

---

## 🧭 Workflow

```text
User Question
     │
     ▼
┌────────────────────────────────────────────────┐
│  LLM #1 (Router)                               │
│  - Decide: Query or Direct Response             │
└────────────────────────────────────────────────┘
     │
     ├──▶ If Query Needed
     │       ▼
     │   Retrieve top-5 similar docs from FAISS
     │
     ▼
┌────────────────────────────────────────────────┐
│  LLM #2 (Generator)                            │
│  - Read retrieved context                      │
│  - Summarize the bug and explain details       │
│  - Output JSON:                                │
│     { "Reported issues": "...",                │
│       "Description": "..." }                   │
└────────────────────────────────────────────────┘
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Pull required Ollama models
Make sure Ollama is installed and running on your machine, then pull the models used by this pipeline:
```bash
ollama pull dengcao/Qwen3-Embedding-8B:Q5_K_M
ollama pull alibayram/Qwen3-30B-A3B-Instruct-2507:latest
```

### 3️⃣ Start the FastAPI server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- `--host 0.0.0.0` → allow connections from any network interface  
- `--port 8000` → serve at port 8000  
- `--reload` → auto-restart when you edit the code (for development)

---

## 💬 Example Query

Use `curl` or any HTTP client to query the `/ask` endpoint:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the issues reported on Pagination?"}'
```

Response (example):
```json
{
  "answer": {
    "Reported issues": "Pagination not updating after filter change",
    "Description": "Users reported that after applying filters, the next page button loads stale data."
  },
  "sources": [
    {
      "metadata": {"file": "bug_report_123.txt"},
      "preview": "Pagination fails when filters applied..."
    }
  ]
}
```

---

## ⚙️ Environment Variables

You can override defaults using environment variables:

| Variable | Default | Description |
|-----------|----------|-------------|
| `OLLAMA_EMBED_MODEL` | `dengcao/Qwen3-Embedding-8B:Q5_K_M` | Embedding model |
| `OLLAMA_CHAT_MODEL` | `alibayram/Qwen3-30B-A3B-Instruct-2507:latest` | Chat/Reasoning model |
| `VECTORSTORE_PATH` | `/workspace/internal/vectorstores/bug_reports_index` | FAISS index directory |

---

## 🧩 Project Structure
```
api_from_notebook/
├── main.py           # FastAPI + LangGraph pipeline
└── requirements.txt  # dependencies
```

---

## 🛠 Notes
- Requires a FAISS index built with the same embedding model.
- Ollama server must be running before startup.
- For production, consider using `gunicorn` with `uvicorn.workers.UvicornWorker` or Docker deployment.

---

**Purpose:** Internal question-answering on bug reports 