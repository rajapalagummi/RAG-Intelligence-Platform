# RAG Intelligence Platform

A dual-chatbot retrieval-augmented generation system built entirely on free, local infrastructure. Combines BM25 keyword search, vector semantic search, and LLM page-index reasoning to solve the core failure mode of standard RAG pipelines — cosine similarity returning zero relevant results on exact financial and legal queries.

## Why this is different from standard RAG

Most RAG implementations chunk documents, embed them, and retrieve by cosine similarity. This works for conceptual queries but fails completely when a user asks something like "net profit Q3 after removing one-time items" — there is zero semantic overlap between the question and the document text, so cosine similarity returns nothing useful.

This platform solves that with a three-layer hybrid retrieval pipeline:

1. **BM25 keyword search** — exact term matching, handles financial and legal queries precisely
2. **Vector semantic search** — ChromaDB cosine similarity for conceptual queries
3. **LLM page-index reasoning** — the model reasons about which section of the document to search before retrieval runs, eliminating irrelevant chunk retrieval

Results from all three layers are merged, deduplicated, and ranked by combined confidence score.

## Architecture

```
User Query
    |
Intent Detection (question / chart / report)
Domain Detection (medical / science / law / finance / general)
    |
HYBRID RETRIEVAL (user documents)
  BM25 keyword search
  + Vector semantic search (ChromaDB)
  + LLM page-index reasoning
  -> Merged, ranked by confidence score
    |
CONTRADICTION DETECTION
  Date extraction -> recency ranking
  Doc-vs-doc conflict detection
  Domain-aware causation chain analysis
  Legal: finds override clauses
  Other: states contradiction, defers to user
    |
PUBLIC CROSS-CHECK (user toggle)
  5 APIs queried in parallel
  Wikipedia, PubMed, ArXiv, SEC EDGAR, GovInfo
  Domain auto-detected from question
    |
Mistral generates answer
Confidence: percentage if above 50%, written qualifier if below
RAGAS self-evaluation per answer
    |
Output: Answer, Inline chart, PDF report, Contradiction panel, Sources
```

## Features

| Feature | Detail |
|---|---|
| Hybrid retrieval | BM25 + vector semantic search + LLM page-index reasoning |
| Contradiction detection | Doc-vs-doc with date extraction and recency scoring |
| Causation analysis | Legal domains: finds override clauses. Others: states contradiction |
| Domain auto-detection | Medical, Science, Law, Finance, General |
| 5 live public APIs | Wikipedia, PubMed, ArXiv, SEC EDGAR, GovInfo (parallel calls) |
| RAGAS evaluation | Faithfulness, Relevancy, Context Precision, Context Recall |
| Chart generation | Plotly pie, bar, and line charts rendered inline from document data |
| PDF reports | Generated on demand with confidence scores and source citations |
| Confidence scoring | Above 50%: percentage shown. Below 50%: written qualifier |
| Dual chatbots | Public (no login) + Personal (login, document upload, cross-check toggle) |
| Zero cost | Mistral via Ollama (local), all public APIs free, ChromaDB local |

## Stack

- **LLM** - Mistral 7B via Ollama (local, no API key required)
- **Vector DB** - ChromaDB (Docker, local storage)
- **Embeddings** - SentenceTransformers all-MiniLM-L6-v2
- **BM25** - Pure Python implementation
- **UI** - Streamlit
- **PDF generation** - fpdf2
- **Auth** - bcrypt password hashing, file-based user store
- **Public APIs** - Wikipedia REST, PubMed E-utilities, ArXiv API, SEC EDGAR, GovInfo

## Setup

Prerequisites: Python 3.12, Docker Desktop, Homebrew (Mac)

```bash
# Step 1 - Install Ollama and pull Mistral
brew install ollama
ollama serve
ollama pull mistral

# Step 2 - Start ChromaDB
docker compose up -d

# Step 3 - Create environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Step 4 - Run
python3 -m streamlit run src/app.py --server.port 8502
```

Open http://localhost:8502

Start order for every session:
1. ollama serve (Terminal 1)
2. docker compose up -d (Terminal 2)
3. python3 -m streamlit run src/app.py --server.port 8502 (Terminal 3)

## Example Queries

**Personal Chatbot - upload your own PDF, DOCX, or TXT documents:**
- "What does section 4 say about liability?"
- "Give me a pie chart of budget allocation by department"
- "How has the policy changed across these documents and why?"
- "What is the net revenue after removing one-time items?"
- "Generate a report on financial performance"

**Public Chatbot - no upload needed:**
- "What are the latest research findings on diabetes prevention?"
- "What does US federal law say about data privacy?"
- "Who is Elon Musk and what companies has he founded?"
- "What are Tesla's latest SEC filings about?"
- "Summarise recent ArXiv papers on large language models"

## Privacy

- User documents are stored locally in ChromaDB and never transmitted to any external server
- User documents are never used to improve or update the public library
- Passwords hashed with bcrypt and never stored in plain text
- No telemetry or external data collection of any kind
