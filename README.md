# 📚 VaultMind - Where Your Documents Think

> A fully **local**, **privacy-first** Retrieval-Augmented Generation (RAG) pipeline that lets you chat with your PDF documents using a local LLM via [Ollama](https://ollama.com/) and [FAISS](https://github.com/facebookresearch/faiss) vector search — no cloud, no API keys required.

---

## ✨ Features

- 📄 **PDF ingestion** — Load any PDF, auto-chunk and embed it into a local FAISS vector store
- 🔍 **Semantic search** — Retrieve the most relevant passages using HuggingFace sentence embeddings
- 🤖 **Local LLM answers** — Generate clear, stepwise answers using Ollama (runs entirely on your machine)
- 💬 **Short-term memory** — Maintains per-user conversation context across queries
- 🚀 **REST API** — Clean FastAPI server so any frontend or chatbot can plug in
- 📊 **Model benchmarking** — Built-in script to compare Ollama models by speed and RAM usage

---

## 🏗️ Architecture

```
User Query
    │
    ▼
FastAPI Server  (/query)
    │
    ├──► Memory Service  →  fetch recent conversation history
    │
    ├──► FAISS Vector DB  →  similarity search (top-k chunks)
    │
    └──► Ollama LLM (phi3:mini)  →  generate grounded answer
                                          │
                                          ▼
                                    Answer returned to user
```

---

## 📁 Project Structure

```
rag-document-assistant/
│
├── src/
│   ├── ingestion/
│   │   └── ingest.py              # PDF loading, chunking, embedding & FAISS storage
│   │
│   ├── retrieval/
│   │   └── retriever.py           # FAISS search + Ollama LLM response generation
│   │
│   ├── memory/
│   │   └── short_term_memory.py   # Fetch & format per-user conversation history
│   │
│   └── api/
│       └── server.py              # FastAPI REST API exposing the RAG pipeline
│
├── scripts/
│   └── benchmark_models.py        # Benchmark multiple Ollama models (speed, RAM)
│
├── tests/
│   └── test_retriever.py          # Unit tests (pytest)
│
├── data/
│   └── sample_docs/               # Place your PDF files here before ingestion
│
├── vector_store/                  # Auto-generated FAISS index (git-ignored)
│
├── .env.example                   # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Document Loading | LangChain + PyPDF |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS (CPU) |
| LLM | Ollama — `phi3:mini` (local, no GPU required) |
| API Framework | FastAPI + Uvicorn |
| Memory | Custom short-term memory service (REST) |
| Testing | pytest |

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- At least 8 GB RAM (for `phi3:mini`)

### 2. Clone the Repository

```bash
git clone https://github.com/iamanimesh11/vaultmind-docs-ai.git
cd vaultmind-docs-ai
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Pull the Ollama Model

```bash
ollama pull phi3:mini
```

### 6. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env if you want to change the model or ports
```

---

## 📥 Step 1 — Ingest Your Document

Place your PDF in `data/sample_docs/` and run:

```bash
python -m src.ingestion.ingest data/sample_docs/your_document.pdf
```

This will:
1. Load and parse the PDF
2. Split it into overlapping chunks (1000 chars, 200 overlap)
3. Generate embeddings using `all-MiniLM-L6-v2`
4. Save the FAISS index to `vector_store/`

---

## 💬 Step 2 — Start the API Server

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8001 --reload
```

The API will be live at `http://localhost:8001`.  
Interactive docs available at `http://localhost:8001/docs`.

---

## 🔎 Step 3 — Query Your Document

Send a POST request to `/query`:

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I submit an IT helpdesk ticket?", "sender_id": "user_123"}'
```

**Response:**
```json
{
  "answer": "🎫 To submit an IT Helpdesk ticket:\n1. Go to the IT portal at ...\n2. Click 'New Ticket' ...\n3. Fill in the details and submit."
}
```

---

## 📊 Benchmarking Ollama Models

Not sure which model to use for your hardware? Run the benchmark:

```bash
python scripts/benchmark_models.py
```

Results are printed to the console and saved to `scripts/benchmark_results.json`.

| Model | RAM Usage | Speed | Best For |
|---|---|---|---|
| `phi3:mini` | ~3 GB | Fast | Low-RAM systems (8 GB) |
| `llama3.2:1b` | ~2 GB | Very Fast | Lightweight responses |
| `llama3.2:3b` | ~4 GB | Moderate | Balanced quality |
| `mistral:7b` | ~6 GB | Slower | Highest quality answers |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🔌 API Reference

### `GET /health`
Liveness check.
```json
{ "status": "ok" }
```

### `POST /query`
Query the RAG pipeline.

**Request Body:**
```json
{
  "query": "string",
  "sender_id": "string"
}
```

**Response:**
```json
{
  "answer": "string",
  "error": "string | null"
}
```

---

## 🛣️ Roadmap

- [ ] Support for multiple document formats (DOCX, TXT, CSV)
- [ ] Persistent multi-turn memory (Redis / SQLite)
- [ ] Streamlit / Gradio frontend UI
- [ ] Docker support for one-command deployment
- [ ] Support for GPU-accelerated FAISS

---
