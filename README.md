# RAG Intelligence Platform

A production-grade dual-chatbot retrieval-augmented generation system built entirely on free, local infrastructure. Combines BM25 keyword search, vector semantic search, and LLM page-index reasoning to solve the core failure mode of standard RAG pipelines — cosine similarity returning zero relevant results on exact financial and legal queries.

## Why this is different from standard RAG

Most RAG implementations chunk documents, embed them, and retrieve by cosine similarity. This works for conceptual queries but fails completely when a user asks something like *"net profit Q3 after removing one-time items"* — there is zero semantic overlap between the question and the document text, so cosine similarity returns nothing useful.

This platform solves that with a three-layer hybrid retrieval pipeline:

1. **BM25 keyword search** — exact term matching, handles financial and legal queries precisely
2. **Vector semantic search** — ChromaDB cosine similarity for conceptual queries
3. **LLM page-index reasoning** — Mistral reasons about which section of the document to search before retrieval runs, borrowed from AlphaGo's planning approach

Results from all three layers are merged, deduplicated, and ranked by combined confidence score.

## Architecture

```
User Query
    ↓
Intent Detection (question / chart / report)
Domain Detection (medical / science / law / finance / general)
    ↓
┌─────────────────────────────────────────┐
│  HYBRID RETRIEVAL (user docs)           │
│  BM25 keyword search                    │
│  + Vector semantic search (ChromaDB)    │
│  + LLM page-index reasoning             │
│  → Merged, ranked by confidence         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  CONTRADICTION DETECTION                │
│  Date extraction → recency ranking      │
│  Doc-vs-doc conflict detection          │
│  Domain-aware causation chain           │
│  (legal: finds override clauses)        │
│  (other: states contradiction, defers)  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  PUBLIC CROSS-CHECK (toggle)            │
│  5 APIs queried in parallel             │
│  Wikipedia · PubMed · ArXiv            │
│  SEC EDGAR · GovInfo US Code           │
│  Domain auto-detected from question     │
└─────────────────────────────────────────┘
    ↓
Mistral generates answer (streaming)
Confidence: ≥50% shows percentage · <50% shows written qualifier
RAGAS self-evaluation (faithfulness, relevancy, precision, recall)
    ↓
Output: Answer · Chart (if requested) · PDF Report (if requested)
        Contradiction panel · Causation chain · Source citations
```

## Key Features

| Feature | Detail |
|---|---|
| Hybrid retrieval | BM25 + Vector + LLM page reasoning merged |
| Contradiction detection | Doc-vs-doc with date extraction and recency scoring |
| Causation analysis | Legal domains: finds override/bypass clauses. Others: states contradiction |
| Domain auto-detection | Medical · Science · Law · Finance · General — detected from question |
| 5 live public APIs | Wikipedia · PubMed · ArXiv · SEC EDGAR · GovInfo (parallel, 3s timeout) |
| RAGAS evaluation | Faithfulness · Relevancy · Context Precision · Context Recall per answer |
| Chart generation | Plotly renders inline — pie, bar, line — LLM extracts data from documents |
| PDF reports | Generated on demand — confidence, sources, contradictions included |
| Confidence scoring | ≥50%: percentage shown · <50%: written qualifier with explanation |
| Dual chatbots | Public (no login) + Personal (login + uploads, toggle cross-check) |
| Zero cost | Mistral via Ollama (local) · All public APIs free · ChromaDB local |

## Stack

- **LLM** — Mistral 7B via Ollama (local, free, no API key)
- **Vector DB** — ChromaDB (Docker, local storage)
- **Embeddings** — SentenceTransformers `all-MiniLM-L6-v2`
- **BM25** — Pure Python implementation (no external library)
- **UI** — Streamlit
- **PDF** — fpdf2
- **Auth** — bcrypt password hashing, file-based user store
- **Public APIs** — Wikipedia REST · PubMed E-utilities · ArXiv API · SEC EDGAR EFTS · GovInfo API

## Setup

```bash
# Step 1 — Install Ollama and pull Mistral
brew install ollama
ollama serve
ollama pull mistral

# Step 2 — Start ChromaDB
docker compose up -d

# Step 3 — Create environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Step 4 — Run
python3 -m streamlit run src/app.py --server.port 8502
```

Open http://localhost:8502

## Test Scenarios

Upload any regulatory, financial, or scientific PDF documents and try:

**Contradiction + Causation (upload two versions of a policy doc):**
```
Until what year is the data retention policy effective?
```
Expected: System detects conflict between documents, identifies the newer one, searches for override clauses, and explains why the change occurred.

**BM25 exact match (financial doc):**
```
What is the net profit for Q3 after removing one-time items?
```
Expected: Exact figure retrieved despite zero semantic overlap with document language.

**Visualisation:**
```
Give me a pie chart of revenue breakdown by product line
```
Expected: Plotly chart rendered inline with data extracted from document.

**Cross-document synthesis:**
```
How has the compliance budget changed between 2019 and 2023 and why?
```
Expected: Figures from both documents synthesised with causal explanation.

**Public knowledge (no upload needed):**
```
What are the latest research findings on diabetes prevention?
```
Expected: PubMed abstracts cited with PMID sources.

## Privacy

- User documents are stored locally in ChromaDB — never transmitted to any server
- User documents are never used to improve the public library
- Passwords hashed with bcrypt — never stored in plain text
- No telemetry, no external data collection

## Resume Framing

**RAG Intelligence Platform** — *Python · Mistral (Ollama) · ChromaDB · BM25 · SentenceTransformers · Streamlit · Docker · Wikipedia · PubMed · ArXiv · SEC EDGAR · GovInfo*

- Engineered hybrid retrieval pipeline combining BM25, vector semantic search and LLM page-index reasoning — achieving accurate extraction on exact financial and legal queries where standard RAG pipelines fail due to zero semantic overlap.
- Designed multi-document contradiction detection engine with date-extraction recency scoring and domain-aware causation chain analysis — reducing manual regulatory review effort by surfacing conflicts and override clauses automatically across documents.
- Built dual-chatbot RAG platform querying 5 live public APIs in parallel across medical, science, law, finance and general domains — delivering sub-5 second responses with RAGAS self-evaluation, inline charts and zero infrastructure cost.
