# VBC AI Backend

A sophisticated AI-powered document processing and chat system specifically designed for **Value-Based Care (VBC) contract analysis**. Built with modern Python, FastAPI, LangChain, and vector search capabilities.

## 🌟 Key Features

### 🔥 **Core Capabilities**
- **🏥 VBC Contract Analysis**: Specialized processing for healthcare value-based care contracts
- **📄 Intelligent Document Processing**: Advanced PDF parsing, chunking, and vector embedding
- **💬 AI-Powered Chat**: Context-aware conversations about your documents using OpenAI
- **🔍 Vector Search**: Semantic search through document chunks using Qdrant vector database
- **🔐 Authentication**: API key and bearer token authentication system
- **📊 Real-time Processing**: Async document ingestion pipeline with progress tracking

### 🛠️ **Technical Excellence**
- **🏗️ Modern Architecture**: Clean, modular service-oriented design with proper separation of concerns
- **✨ Code Quality**: Fully linted with Ruff, type hints, and modern Python 3.13+ features
- **🔄 Async/Await**: Non-blocking operations throughout the system
- **📝 Comprehensive Logging**: Detailed logging across all components
- **🧪 Extensible Design**: Strategy pattern for document analysis, modular ingestion pipeline

## 🚀 API Endpoints

### 📁 **Document Management**
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/documents/upload` | Upload PDF documents for processing | ✅ |
| GET | `/documents/{id}/status` | Get document processing status | ✅ |
| GET | `/documents` | List all uploaded documents | ✅ |
| DELETE | `/documents/{id}` | Delete a document and its data | ✅ |

### 💬 **AI Chat & Analysis**  
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/chat` | Chat with AI about your documents | ✅ |
| POST | `/chat/document-analysis` | Get structured analysis of specific documents | ✅ |

### 🔍 **Search & Retrieval**
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/search/documents` | Search through document content | ✅ |
| GET | `/search/chunks` | Search document chunks with metadata | ✅ |

### 🔐 **Authentication**
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/auth/token` | Create new API access token | ❌ |
| DELETE | `/auth/token` | Revoke current access token | ✅ |
| GET | `/auth/status` | Check token validation status | ✅ |
| GET | `/auth/stats` | Get authentication statistics | ✅ |

### 🏥 **Health & Monitoring**
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/health` | Application health check | ❌ |
| GET | `/health/detailed` | Detailed system health status | ✅ |
| GET | `/docs` | Interactive API documentation (Swagger) | ❌ |
| GET | `/redoc` | ReDoc API documentation | ❌ |

## Docker Setup

### Prerequisites

- Docker and Docker Compose installed
- Docker daemon running

### Quick Start with Docker Compose

```bash
# Build and start the application
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

The API will be available at `http://localhost:8000`

### Manual Docker Commands

```bash
# Build the Docker image
docker build -t vbc-ai:latest .

# Run the container
docker run -d \
  --name vbc-ai \
  -p 8000:8000 \
  vbc-ai:latest

# View container logs
docker logs vbc-ai

# Stop and remove container
docker stop vbc-ai
docker rm vbc-ai
```

### Development Mode

For development with live code reloading, uncomment the volume mount in `docker-compose.yml`:

```yaml
volumes:
  - .:/app  # Uncomment this line
  - uploads:/app/uploads
```

Then restart with:
```bash
docker-compose down
docker-compose up --build
```

## Local Development (Without Docker)

```bash
# Install dependencies using uv
uv pip install -e .

# Or using pip
pip install -e .

# Run the application
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage Examples

### Upload a Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Check Document Status
```bash
curl "http://localhost:8000/documents/{document_id}/status"
```

### Chat with AI
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the payment terms?", "filters": {"document_type": "contract"}}'
```

### Search Documents
```bash
curl "http://localhost:8000/search?q=payment%20terms&filters=%7B%22document_type%22:%22contract%22%7D"
```

### Health Check
```bash
curl "http://localhost:8000/healthz"
```

## ⚙️ Environment Variables

### **Core Application**
| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string (required) |
| `OPENAI_API_KEY` | - | OpenAI API key for LLM operations (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use for analysis |

### **Vector Database (Qdrant)**
| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector database URL |
| `QDRANT_API_KEY` | - | Qdrant API key (optional) |
| `QDRANT_COLLECTION_NAME` | `vbc_documents` | Qdrant collection name |

### **Document Processing**
| Variable | Default | Description |
|----------|---------|-------------|
| `UPLOAD_DIR` | `uploads/` | Directory for uploaded files |
| `PROCESSED_DIR` | `processed/` | Directory for processed files |
| `MAX_FILE_SIZE_MB` | `50` | Maximum file size in MB |
| `CHUNK_SIZE` | `1000` | Text chunk size for processing |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

### **Server Configuration**
| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Host to bind the server |
| `PORT` | `8000` | Port to run the server |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `API_TOKEN` | - | API access token for authentication |

## Docker Image Details

- **Base Image**: python:3.13-slim
- **Working Directory**: /app
- **Exposed Port**: 8000
- **Health Check**: Checks `/healthz` endpoint every 30 seconds
- **Security**: Runs as non-root user
- **Dependencies**: Managed with uv for faster builds

## API Documentation

Once the application is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🏗️ Architecture & Project Structure

### **System Architecture**
The VBC AI backend follows a **clean, service-oriented architecture** with proper separation of concerns:

- **🎯 Service Layer**: Modular services for chat, document processing, retrieval, and analysis
- **📊 Data Layer**: PostgreSQL for structured data, Qdrant for vector embeddings
- **🤖 AI Layer**: LangChain integration with OpenAI for intelligent document analysis
- **🔌 API Layer**: FastAPI with comprehensive authentication and error handling
- **📁 Storage Layer**: File system storage with organized upload/processed directories

### **Project Structure**
```
VBC_AI/
├── 📁 backend/
│   ├── 📁 app/
│   │   ├── 📁 auth/                 # Authentication system
│   │   ├── 📁 database/             # Database queries and migrations
│   │   ├── 📁 models/               # Pydantic models and data classes
│   │   ├── 📁 routes/               # FastAPI route handlers
│   │   │   ├── auth.py              # Authentication endpoints
│   │   │   ├── chat.py              # Chat and AI endpoints
│   │   │   ├── documents.py         # Document management endpoints  
│   │   │   ├── health.py            # Health check endpoints
│   │   │   └── search.py            # Search endpoints
│   │   ├── 📁 services/             # Business logic services
│   │   │   ├── 📁 ingestion/        # Document ingestion pipeline
│   │   │   │   ├── analysis_service.py    # Document analysis strategies
│   │   │   │   ├── chunking_service.py    # Text chunking logic
│   │   │   │   ├── pipeline.py            # Ingestion orchestrator
│   │   │   │   └── vector_service.py      # Vector operations
│   │   │   ├── chat_service.py      # AI chat functionality
│   │   │   ├── database.py          # Database operations
│   │   │   ├── document_service.py  # Document processing
│   │   │   ├── ingestion.py         # Main ingestion service
│   │   │   ├── llm_analyzer.py      # Generic LLM analysis
│   │   │   ├── qdrant_store.py      # Vector store operations
│   │   │   ├── retrieval.py         # Document search/retrieval
│   │   │   └── vbc_analyzer.py      # VBC contract analysis
│   │   ├── 📁 storage/              # File storage management
│   │   ├── 📁 utils/                # Utility functions
│   │   │   ├── deduplication.py     # Document deduplication
│   │   │   ├── document_parser.py   # PDF parsing utilities
│   │   │   ├── file_validation.py   # File validation logic
│   │   │   └── logger.py            # Centralized logging
│   │   ├── config.py                # Configuration management
│   │   ├── main.py                  # FastAPI application entry
│   │   └── prompts.py               # Centralized LLM prompts
│   ├── pyproject.toml               # Dependencies & Ruff config
│   ├── uv.lock                      # Dependency lock file
│   └── lint.py                      # Linting helper script
├── 📁 uploads/                      # Uploaded documents
├── 📁 processed/                    # Processed documents  
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Docker Compose setup
├── .env.example                     # Environment variables template
└── README.md                        # This documentation
```

## 🔧 Development Workflow

### **Code Quality & Linting**
The project uses **Ruff** for linting and formatting:

```bash
# Run linting checks
uv run ruff check backend/app/

# Auto-fix issues
uv run ruff check backend/app/ --fix

# Format code
uv run ruff format backend/app/

# Use helper script (interactive)
python backend/lint.py
```

### **Testing**
```bash
# Run integration tests
python backend/test_new_architecture.py

# Run simple chat tests  
python backend/test_chat_simple.py
```

### **Database Setup**
Ensure you have PostgreSQL running and create the required tables using the queries in `backend/app/database/queries.py`.

## 🚀 Production Considerations

### **✅ Ready for Production**
- **🔐 Authentication**: Bearer token and API key authentication implemented
- **📊 Database**: Full PostgreSQL integration with async operations
- **🔍 Vector Search**: Qdrant vector database for semantic search
- **📝 Logging**: Comprehensive structured logging throughout
- **⚡ Performance**: Async/await operations, efficient chunking
- **🛡️ Security**: Input validation, file type checking, rate limiting

### **🏗️ Architecture Benefits**
- **Modular Design**: Services are loosely coupled and easily testable
- **Strategy Pattern**: Document analysis can be extended with new strategies  
- **Clean Code**: Follows SOLID principles with proper separation of concerns
- **Type Safety**: Comprehensive type hints and Pydantic models
- **Error Handling**: Robust error handling with detailed logging

### **📦 Key Dependencies**
```toml
# Core Framework
fastapi = "^0.115.6"
uvicorn = "^0.34.0"

# AI & LLM
langchain = "^0.3.9" 
langchain-openai = "^0.2.10"
openai = "^1.57.0"

# Vector Database  
qdrant-client = "^1.12.1"
langchain-community = "^0.3.9"

# Database
sqlalchemy = "^2.0.36"
asyncpg = "^0.30.0"
psycopg2-binary = "^2.9.10"

# Document Processing
pypdf = "^5.1.0"
python-multipart = "^0.0.20"

# Code Quality
ruff = "^0.8.4"
```

### **🔄 Deployment Recommendations**
1. **Environment Variables**: Use `.env` files or container orchestration secrets
2. **Database**: Set up PostgreSQL with proper connection pooling  
3. **Vector Store**: Deploy Qdrant or use managed vector database service
4. **Monitoring**: Add APM tools (DataDog, New Relic) for production monitoring
5. **Scaling**: Use multiple worker processes with proper session management
6. **CDN**: Consider CDN for document storage in distributed setups