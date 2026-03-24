# Redmine RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that searches Redmine issue history and structured clinical/domain data, built with Flask + ChromaDB + Gemini.

## Features

- **Dual-source routing**: Automatically routes queries between Redmine issues and secondary structured data (e.g., clinical records) using keyword detection + vector similarity comparison
- **Hybrid search**: Dense vector search + BM25 (Redmine only), merged with RRF (Reciprocal Rank Fusion)
- **Multi-turn conversation**: Flask server-side session (filesystem) with ChromaDB-backed conversation history
- **Image attachment support**: Proxies Redmine image attachments with overkill guard (prevents hallucinating metrics from image-only issues)
- **CRF statistics**: Python code execution via Gemini for aggregated statistics and chart generation
- **Diagram generation**: PaperBanana v2 integration (SSE proxy)
- **Observability**: Langfuse tracing for embedding / retrieval / LLM generation spans

## Tech Stack

| Component | Library |
|-----------|---------|
| Backend | Flask 3.0.0 + Gunicorn 21.2.0 |
| LLM | Google Gemini 2.5 Pro (`google-genai`) |
| Embedding | `models/gemini-embedding-001` |
| Vector DB | ChromaDB 1.3.5 |
| Hybrid Search | `rank-bm25` + RRF |
| Session | `flask-session` (filesystem) |
| Observability | Langfuse (optional) |

## Project Structure

```
.
├── docker-compose.yml
├── .env.example
├── chatbot/
│   ├── Dockerfile
│   ├── requirement.txt
│   ├── templates/
│   │   └── chat.html
│   └── src/
│       ├── app.py                    # Flask app, routing, session
│       ├── rag_engine.py             # RedmineRAG class, query() orchestration
│       ├── prompts.py                # Prompt templates (redmine / crf / general)
│       ├── config/
│       │   ├── constants.py          # Thresholds, staff lists, BM25 config
│       │   ├── patterns.py           # Regex patterns for routing
│       │   ├── diagram_config.py     # PaperBanana config
│       │   └── gunicorn_config.py
│       └── utils/
│           ├── rag_utils.py          # Embedding, search query builder, context formatter
│           ├── rag_engine_helpers.py # Search, cutoff, post-process, answer generation
│           ├── crf_statistics.py     # Aggregated statistics for structured data
│           └── paperbanana_client.py # PaperBanana SSE client
```

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/yourusername/redmine-rag-chatbot.git
cd redmine-rag-chatbot

cp .env.example .env
# Edit .env — fill in GEMINI_API_KEY, REDMINE_URL, REDMINE_API_KEY
```

### 2. Prepare Vector DB

Place your ChromaDB data under `./vectordb/`:

```
vectordb/
├── chroma_db_v0.2.0/        # Redmine issues collection
├── crf_data/
│   └── chroma_db_v0.3.0/    # Secondary structured data (optional)
└── conversation_db/          # Auto-created on first run
```

You need to build the ChromaDB collections from your data source. See the [Vector DB section](#vector-db-setup) below.

### 3. Run with Docker Compose

```bash
docker compose up -d
# Access at http://localhost:8080
```

### 4. Local development

```bash
pip install -r chatbot/requirement.txt

export GEMINI_API_KEY=your_key
export VECTORDB_PATH=./vectordb/chroma_db_v0.2.0
export COLLECTION_NAME=redmine_issues_raw_v4
export REDMINE_URL=https://your-redmine.example.com
export REDMINE_API_KEY=your_redmine_api_key

cd chatbot
gunicorn --config src/config/gunicorn_config.py "src.app:app"
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `REDMINE_URL` | Yes | Redmine instance URL |
| `REDMINE_API_KEY` | Yes | Redmine API key (for image proxy) |
| `SECRET_KEY` | Yes | Flask session secret (change in production) |
| `VECTORDB_PATH` | No | Path to Redmine ChromaDB (default: `/vectordb/chroma_db_v0.2.0`) |
| `COLLECTION_NAME` | No | Redmine collection name (default: `redmine_issues_raw_v4`) |
| `CRF_VECTORDB_PATH` | No | Path to secondary ChromaDB (optional) |
| `CRF_COLLECTION_NAME` | No | Secondary collection name (optional) |
| `PORT` | No | Server port (default: `8080`) |
| `GUNICORN_WORKERS` | No | Worker count (default: `cpu*2+1`) |
| `PAPERBANANA_BASE_URL` | No | PaperBanana service URL (diagram feature) |
| `LANGFUSE_HOST` | No | Langfuse server URL (observability) |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Chat UI |
| POST | `/chat` | RAG query |
| POST | `/reset` | Reset session history |
| GET | `/health` | Health check |
| GET | `/users` | List users with history |
| DELETE | `/users/<user_name>` | Delete user history |
| POST | `/visualize` | Rewrite for diagram generation |
| POST | `/pb-start` | Start PaperBanana task |
| GET | `/pb-stream/<task_id>` | SSE progress proxy |
| GET | `/pb-image/<task_id>` | Result image (base64) |
| DELETE | `/pb-cancel/<task_id>` | Cancel task |

## Vector DB Setup

### Redmine Issues

Use an Airflow DAG (or any scheduler) to fetch issues from the Redmine API and upsert into ChromaDB:

```python
import chromadb
from google import genai

client = chromadb.PersistentClient(path="./vectordb/chroma_db_v0.2.0")
col = client.get_or_create_collection("redmine_issues_raw_v4")

# For each Redmine issue:
# 1. Fetch via Redmine REST API
# 2. Build document text from subject + description + journals
# 3. Extract image attachment metadata
# 4. Embed with Gemini embedding-001
# 5. Upsert into collection with metadata

genai_client = genai.Client(api_key="YOUR_KEY")
embedding = genai_client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=document_text,
    config={"task_type": "RETRIEVAL_DOCUMENT"}
).embeddings[0].values

col.upsert(
    ids=[str(issue_id)],
    embeddings=[embedding],
    documents=[document_text],
    metadatas=[{
        "issue_id": issue_id,
        "subject": subject,
        "attachment_ids": json.dumps(image_attachment_ids),
        # ... other metadata
    }]
)
```

### Metadata Schema (Redmine)

| Field | Type | Description |
|-------|------|-------------|
| `issue_id` | int | Redmine issue ID |
| `subject` | str | Issue title |
| `author` | str | Issue author |
| `created_on` | str | ISO datetime |
| `updated_on` | str | ISO datetime |
| `attachment_ids` | str (JSON) | Image attachment IDs |
| `attachment_filenames` | str (JSON) | `{id: filename}` |
| `attachment_urls` | str (JSON) | `{id: content_url}` |

## Observability (Langfuse)

When `LANGFUSE_*` env vars are set, each request creates a trace with:

```
Trace: rag-query
  metadata: engine_name, use_case, route_reason, status
  ├── Span: embedding
  ├── Span: retrieval  (doc_count)
  └── Span: llm-generation  (model, prompt, answer)
```

Self-host Langfuse with Docker: [langfuse/langfuse](https://github.com/langfuse/langfuse)

## Changelog

### v0.3.3
- flask-session (filesystem) — eliminates cookie size limit for sessions with many images
- Image attachment metadata in ChromaDB (attachment_ids, filenames, urls)
- Image overkill guard: `HAS_IMAGE_ATTACHMENTS` / `TEXT_METRICS_PRESENT` tags in context
- Langfuse tracing integration (embedding / retrieval / llm-generation spans)
- DAG schedule: Wednesday 07:30 KST

### v0.3.2
- Redmine image proxy (`/redmine-image/<attachment_id>`)
- Image thumbnails in source references (`[IMG n]`)

### v0.3.1
- Dual-source routing (Redmine + CRF) with vector similarity fallback
- BM25 hybrid search with RRF for Redmine
- CRF statistics with Gemini code execution

## License

MIT
