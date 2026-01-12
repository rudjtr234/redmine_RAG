# 🤖 Multi-Source RAG Chatbot

An intelligent RAG (Retrieval-Augmented Generation) chatbot system that integrates multiple data sources with automatic routing capabilities.

## 🌟 Key Features

### 🎯 Intelligent Multi-Source Routing
- **Automatic Database Selection**: Routes queries to the appropriate data source based on content analysis
- **Dual Routing Strategy**:
  - Fast keyword-based routing for explicit queries
  - Vector similarity comparison for ambiguous questions
- **Context-Aware**: Maintains conversation context for follow-up questions

### 💬 Advanced Conversation Management
- **Multi-turn Dialogue**: Remembers conversation history for contextual responses
- **Session Management**: Per-user conversation tracking and history
- **Vector-based History Search**: Retrieves relevant past conversations to improve answers

### 📊 Data Source Integration
- **Project Management Data**: Integrates with issue tracking systems
- **Research Data**: Supports structured research data with statistical analysis
- **Extensible Architecture**: Easy to add new data sources

### 📈 Statistical Analysis & Visualization
- **Automated Statistics**: Python-based calculation for numerical queries
- **Chart Generation**: Automatic visualization using Gemini Code Execution
- **Multi-field Filtering**: Complex query support with multiple conditions

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
┌────────▼────────────────────────────────┐
│     Intelligent Router                  │
│  (Keyword + Vector Similarity)          │
└────────┬────────────────────────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼───┐
│ DB 1 │  │ DB 2 │  ... (Extensible)
└───┬──┘  └──┬───┘
    │         │
    └────┬────┘
         │
┌────────▼─────────┐
│  Gemini LLM      │
│  (Answer Gen)    │
└────────┬─────────┘
         │
┌────────▼─────────┐
│   Response       │
└──────────────────┘
```

## 🛠️ Tech Stack

- **Backend**: Flask (Python 3.11)
- **Vector Database**: ChromaDB
- **LLM**: Google Gemini 2.5 Pro
- **Embedding**: Gemini Embedding 001
- **Server**: Gunicorn
- **Containerization**: Docker & Docker Compose
- **Orchestration**: Kubernetes (Deployment ready)

## 📁 Project Structure

```
.
├── chatbot/
│   ├── src/
│   │   ├── app.py                  # Flask application & routing logic
│   │   ├── rag_engine.py          # Core RAG engine
│   │   ├── prompts.py             # Prompt templates
│   │   ├── config/
│   │   │   ├── constants.py       # Configuration constants
│   │   │   ├── patterns.py        # Regex patterns for routing
│   │   │   └── gunicorn_config.py # Gunicorn settings
│   │   └── utils/
│   │       ├── rag_utils.py           # RAG helper functions
│   │       ├── rag_engine_helpers.py  # Query processing helpers
│   │       └── crf_statistics.py      # Statistical analysis module
│   ├── templates/
│   │   └── chat.html              # Web UI
│   ├── Dockerfile                 # Docker image definition
│   └── requirement.txt           # Python dependencies
├── docker-compose.yml            # Docker Compose configuration
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))
- Vector database (ChromaDB) with your data

### Installation

1. **Clone the repository**
\`\`\`bash
git clone <repository-url>
cd rag-chatbot
\`\`\`

2. **Set up environment variables**
\`\`\`bash
cp .env.example .env
# Edit .env and add your configurations
\`\`\`

3. **Prepare your vector database**
\`\`\`bash
# Place your ChromaDB data in ./vectordb/
mkdir -p vectordb/chroma_db_v0.2.0
# Copy your ChromaDB files here
\`\`\`

4. **Run with Docker Compose**
\`\`\`bash
docker-compose up -d
\`\`\`

5. **Access the chatbot**
\`\`\`
http://localhost:<PORT>
\`\`\`

### Local Development

\`\`\`bash
cd chatbot

# Install dependencies
pip install -r requirement.txt

# Set environment variables
export GEMINI_API_KEY=your_api_key_here
export VECTORDB_PATH=/path/to/vectordb
export COLLECTION_NAME=your_collection_name

# Run the application
python src/app.py

# Or with Gunicorn
gunicorn --config src/config/gunicorn_config.py src.app:app
\`\`\`

## 📡 API Endpoints

### \`POST /chat\`
Main chatbot endpoint with automatic routing

**Request:**
\`\`\`json
{
  "question": "What is the latest model performance?",
  "user_name": "user123",
  "top_k": 5
}
\`\`\`

**Response:**
\`\`\`json
{
  "answer": "The latest model achieved 95% accuracy...",
  "sources": [
    {
      "issue_id": "123",
      "subject": "Model Performance Update",
      "url": "https://your-server/issues/123"
    }
  ],
  "question": "What is the latest model performance?"
}
\`\`\`

### \`GET /health\`
Health check endpoint

### \`POST /reset\`
Reset conversation history

### \`GET /users\`
List all users with conversation history

### \`DELETE /users/<user_name>\`
Delete user and their conversation history

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| \`GEMINI_API_KEY\` | Google Gemini API key | (required) |
| \`VECTORDB_PATH\` | Path to ChromaDB database | \`/vectordb/chroma_db_v0.2.0\` |
| \`COLLECTION_NAME\` | ChromaDB collection name | \`your_collection_name\` |
| \`EMBEDDING_MODEL\` | Embedding model type | \`gemini\` |
| \`PORT\` | Server port | \`<PORT>\` |
| \`GUNICORN_WORKERS\` | Number of worker processes | \`6\` |
| \`LOG_LEVEL\` | Logging level | \`info\` |

For multiple data sources, add additional configuration with prefix (e.g., \`CRF_VECTORDB_PATH\`).

## 🎯 Use Cases

### 1. Technical Documentation Search
Query project documentation, experiment results, and technical specifications.

### 2. Research Data Analysis
Analyze structured research data with automatic statistical calculations and visualization.

### 3. Multi-domain Knowledge Base
Integrate multiple knowledge bases with automatic query routing.

### 4. Conversational AI Assistant
Intelligent assistant that remembers context and handles follow-up questions.

## 🔒 Security

- Store API keys in environment variables
- Use \`.env\` files for local development
- Implement authentication for production
- Use secret management systems (Kubernetes Secrets, AWS Secrets Manager)
- Regular security updates

## 📊 Performance

- **Response Time**: < 3 seconds for typical queries
- **Concurrent Users**: Scales with Gunicorn workers
- **Vector Search**: Optimized with ChromaDB HNSW indexing

## 🚢 Deployment

### Docker Deployment

\`\`\`bash
# Build image
docker build -t rag-chatbot:latest ./chatbot

# Run container
docker run -d -p <PORT>:<PORT> -v ./vectordb:/vectordb -e GEMINI_API_KEY=your_key --name rag-chatbot rag-chatbot:latest
\`\`\`

### Kubernetes

Kubernetes manifests included for production deployment.

## 📈 Roadmap

- [ ] Support for more LLM providers
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Voice input/output capabilities
- [ ] Additional data source integrations

## 📝 License

MIT License

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
