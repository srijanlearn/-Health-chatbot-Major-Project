# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an LLM-powered document Q&A system (HackRx 6.0 submission) that processes insurance policy documents and answers questions with high precision. The system uses a dual-path approach: retrieval-based search for general context questions and full-context search for specific fact queries.

**Tech Stack:**
- FastAPI backend with async/await support
- LangChain framework with Google Gemini models (gemini-1.5-pro, gemini-1.5-flash)
- ChromaDB vector store with HuggingFace embeddings (all-MiniLM-L6-v2)
- Unstructured library for document parsing with OCR support
- Docker deployment (configured for Hugging Face Spaces)

## Development Commands

### Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```powershell
# Run locally (development)
uvicorn app.main:app --reload --port 8000

# Run with Docker
docker build -t turingntesla .
docker run -p 80:80 turingntesla
```

**Important:** The application requires `GOOGLE_API_KEY` environment variable. Create a `.env` file with:
```
GOOGLE_API_KEY=your_api_key_here
```

### API Testing
```powershell
# The main endpoint is POST /hackrx/run
# Access API docs at http://localhost:8000/docs (or http://localhost:80/api/v1/docs in Docker)
```

## Architecture

### Request Flow
1. **Document Download** (`main.py`): Downloads PDF from URL, assigns UUID as document ID
2. **Document Ingestion** (`ingestion.py`): Processes PDF → creates vector embeddings → builds ParentDocumentRetriever
3. **Question Routing** (`main.py`): Router LLM (gemini-flash) classifies each question as "Specific Fact" or "General Context"
4. **Dual-Path Processing**:
   - **Path A (Specific Fact)**: Uses full document text with gemini-pro for precise extraction of numbers/dates/names
   - **Path B (General Context)**: Uses RAG with retriever + gemini-pro for broader summaries
5. **Response Generation**: Returns array of answers matching the input questions

### Key Components

**app/main.py**
- FastAPI app with lifespan management (loads LLM models on startup)
- Implements POST `/hackrx/run` endpoint
- Contains router chain and final Q&A prompt template
- Router classifies questions to determine processing path

**app/ingestion.py**
- Handles PDF processing with Unstructured loader (includes OCR)
- Implements ParentDocumentRetriever pattern:
  - Parent chunks: 2000 chars (for context)
  - Child chunks: 400 chars (for precise vector search)
- Uses Chroma vector store with persistent storage in `./db/<document_id>/`
- Returns both retriever and full_text for dual-path approach

**app/parsers.py**
- Contains utility function to extract "Table of Benefits" from insurance documents
- Currently unused in main flow but available for structured data extraction

### Data Storage
- `./downloaded_files/`: Temporary PDF storage (UUID-named files)
- `./db/<document_id>/`: ChromaDB persistence per document

### Critical Design Patterns
1. **ParentDocumentRetriever**: Searches on small chunks (400 chars) but returns larger parent chunks (2000 chars) for better context
2. **Router-based Query Classification**: Routes queries to optimal processing strategy before retrieval
3. **Metadata Filtering**: Uses `filter_complex_metadata()` to avoid ChromaDB serialization issues
4. **In-Memory Docstore**: Parent documents stored in memory (InMemoryStore) for fast retrieval

### Prompt Engineering Notes
The system uses strict instructions to enforce:
- Context-only answers (no hallucination)
- Yes/No format for objective questions
- Synthesizing scattered information (e.g., waiting periods by cross-referencing procedure lists with category tables)
- Ruthless conciseness except for definitional questions

## Environment Configuration

Required:
- `GOOGLE_API_KEY`: Google Gemini API key

Paths (auto-created):
- `DOWNLOAD_PATH`: `./downloaded_files`
- `DB_BASE_PATH`: `./db`

## Docker Deployment

The Dockerfile is configured for Hugging Face Spaces deployment with:
- Root path: `/api/v1`
- Port: 80
- Base image: `python:3.11-slim`
