# VBC AI API

A FastAPI-based document processing and AI chat API with full Docker support.

## Features

- **Document Upload**: Upload documents and get unique document IDs
- **Document Processing**: Track processing status (queued → processing → done → error)
- **Content Extraction**: Extract normalized JSON from processed documents with PHI redaction
- **AI Chat**: Chat with AI about documents with optional filters
- **Search**: Search through document chunks with metadata
- **Health & Metrics**: Health checks and application metrics
- **OpenAPI/Swagger**: Auto-generated API documentation at `/docs`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a document, returns document_id |
| GET | `/documents/{id}/status` | Get document processing status |
| GET | `/documents/{id}/extracted` | Get extracted document content (when done) |
| POST | `/chat` | Chat with AI about documents |
| GET | `/search` | Search through document chunks |
| GET | `/healthz` | Health check endpoint |
| GET | `/metrics` | Application metrics |
| GET | `/docs` | Interactive API documentation |

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

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| HOST | 0.0.0.0 | Host to bind the server |
| PORT | 8000 | Port to run the server |
| PYTHONUNBUFFERED | 1 | Ensure Python output is sent straight to terminal |

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

## Project Structure

```
VBC_AI/
├── main.py              # FastAPI application
├── pyproject.toml       # Python dependencies
├── uv.lock             # Dependency lock file
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── .dockerignore       # Docker ignore patterns
└── README.md           # This file
```

## Notes

- The current implementation uses in-memory storage for demo purposes
- In production, replace with a proper database (PostgreSQL, MongoDB, etc.)
- Add authentication and authorization as needed
- Implement actual document processing and AI logic
- Configure logging and monitoring for production use