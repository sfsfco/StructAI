# StructAI – Scalable AI Platform for Structured Data Extraction
### Project Under Construction.
**StructAI** is a production-inspired backend platform that transforms unstructured text (reports, notes, contracts, emails) into structured, machine-readable data.
It combines **OpenAI-powered LLMs** with **Google LangExtract** for reliable information extraction, and is built with modern, scalable backend architecture: **FastAPI**, **FAISS**, **Redis**, **PostgreSQL**, background workers, and **Docker Compose**.

> *“Cooking up a scalable AI system — fresh out of the oven.”* 🍰

---

## ✨ Features

* 🧠 **AI-Powered Extraction** – Structured data extraction using OpenAI LLMs + Google LangExtract
* 🔎 **Semantic Retrieval** – FAISS vector search for efficient document retrieval
* ⚡ **Async Processing** – Background workers for heavy ingestion & embedding tasks
* 🧰 **Caching** – Redis cache to reduce latency and LLM costs
* 🗄️ **Persistence** – PostgreSQL for metadata and job tracking
* 🐳 **Dockerized** – One-command local setup with Docker Compose
* 📈 **Scalable by Design** – Stateless API, isolated services, horizontal scaling ready
* 🧪 **Testable Architecture** – Unit, integration, and E2E testing strategy

---

## 🏗️ Architecture (High-Level)

```text
[ Client ]
    |
    v
[ FastAPI API Gateway ]
    |
    +--> [ LangExtract Service ] ---> [ OpenAI API (External) ]
    |
    +--> [ Embedding Service ] ---> [ FAISS Vector Store ]
    |
    +--> [ Redis Cache & Queue Broker ]
    |
    +--> [ PostgreSQL ]
    |
    +--> [ Background Workers ]
```

**Key Design Principles:**

* Stateless API → horizontal scaling
* AI provider abstracted behind a service layer
* Async workers → API remains responsive under load
* Vector search → scalable retrieval over large corpora

---

## 🧰 Tech Stack

* **Backend:** Python 3.11, FastAPI
* **LLM:** OpenAI API (external)
* **Extraction:** Google LangExtract
* **Vector Store:** FAISS
* **Cache & Queue Broker:** Redis
* **Database:** PostgreSQL
* **Workers:** Celery or RQ
* **Containerization:** Docker, Docker Compose
* **Testing:** pytest

---

## 📦 Project Structure

```text
ai-extraction-platform/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── api/                    # Routes
│   ├── services/               # LLM, LangExtract, FAISS, Redis
│   ├── workers/                # Background tasks
│   ├── db/                     # DB session & models
│   └── schemas/                # Pydantic schemas
├── Dockerfile                  # API container
├── Dockerfile.worker           # Worker container
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Getting Started (Docker)

### 1️⃣ Prerequisites

* Docker & Docker Compose
* OpenAI API Key

### 2️⃣ Setup Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here

POSTGRES_USER=ai_user
POSTGRES_PASSWORD=ai_pass
POSTGRES_DB=ai_db

REDIS_HOST=redis
REDIS_PORT=6379
```

> ⚠️ Never commit your real API key. `.env` is ignored by git.

### 3️⃣ Run the Stack

```bash
docker-compose up --build
```

API will be available at:

```
http://localhost:8000
```

Health check:

```
GET /health
```

---

## 🔌 API Examples

### Index a Document (Async)

```http
POST /documents/index
{
  "document_id": "doc_123",
  "text": "Long unstructured document..."
}
```

### Extract Structured Data

```http
POST /extract
{
  "document_id": "doc_123",
  "instructions": "Extract names, dates, and risks"
}
```

**Response:**

```json
{
  "entities": [
    {
      "name": "John Doe",
      "date": "2024-01-12",
      "risk": "Payment delay"
    }
  ]
}
```

---

## 🧪 Testing Strategy

### Unit Tests

* LLM client (mock OpenAI responses)
* LangExtract service
* FAISS vector store wrapper
* Redis cache layer

### Integration Tests

* FastAPI endpoints
* PostgreSQL + Redis integration
* FAISS indexing & retrieval

### End-to-End (E2E)

* Ingest document → embed → index → extract → validate response

```bash
pytest
```

---

## 📈 Scalability & System Design

* **Stateless API** → scale horizontally behind a load balancer
* **Worker Pool** → scale ingestion and embedding independently
* **Redis Cache** → reduce repeated LLM calls and latency
* **FAISS Abstraction** → can be replaced by a managed vector DB in production
* **Rate Limiting & Backpressure** → protect LLM costs and system stability
* **Future Kubernetes Deployment** → production-ready migration path

> *“Ingestion, retrieval, and extraction pipelines are decoupled to enable independent scaling under high load.”*

---

## 🔐 Security Notes

* Secrets managed via environment variables
* No API keys committed to the repository
* Rate limiting on API endpoints
* Input size limits to prevent abuse
* Logs exclude sensitive content

---

## 🛣️ Roadmap / Future Improvements

* Replace FAISS with managed vector DB (e.g., Pinecone, Weaviate)
* Add Kubernetes manifests
* Observability: metrics, tracing, dashboards
* Multi-LLM provider support
* Streaming ingestion & real-time extraction

---

## 🧑‍🍳 Why This Project?

This project goes beyond simple “LLM calls” and demonstrates:

* Real-world **AI system design**
* **Production-style backend architecture**
* **Scalability, async processing, caching, and vector search**
* Practical integration of **Google LangExtract** with LLMs

