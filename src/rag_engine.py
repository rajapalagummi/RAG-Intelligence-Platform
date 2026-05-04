"""
RAG Core Engine v2
==================
Hybrid retrieval: BM25 keyword + Vector semantic + LLM page reasoning
Contradiction detection: doc-vs-doc with recency awareness
Causation analysis: multi-hop reasoning for legal/regulatory domains
Confidence scoring: percentage ≥50%, written qualifier <50%
Model: Phi-3 Mini via Ollama (faster than Mistral on M4)
"""

import re
import json
import math
import logging
import requests
from pathlib import Path
from typing import Optional
from collections import Counter

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"
CHROMA_HOST  = "localhost"
CHROMA_PORT  = 8000

# Legal/regulatory domains that get causation chain treatment
LEGAL_DOMAINS = {"legal", "regulatory", "contract", "compliance", "law", "policy"}

# ── SINGLETON EMBEDDER ─────────────────────────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def get_chroma():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


# ════════════════════════════════════════════════════════════
# OLLAMA LLM CALL
# ════════════════════════════════════════════════════════════

def llm_call(prompt: str, system: str = "", max_tokens: int = 512) -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    try:
        r = requests.post(OLLAMA_URL, json={
            "model":  OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,
                "num_ctx":     2048,
                "top_k":       10,
            }
        }, timeout=30)
        return r.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return ""


# ════════════════════════════════════════════════════════════
# BM25 KEYWORD SEARCH (pure Python, no extra library)
# Handles exact financial/legal term queries that semantic fails on
# ════════════════════════════════════════════════════════════

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.corpus: list[list[str]] = []
        self.raw:    list[dict]      = []
        self.df:     Counter         = Counter()
        self.avgdl:  float           = 0.0
        self.N:      int             = 0

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def index(self, chunks: list[dict]):
        self.raw    = chunks
        self.corpus = [self.tokenize(c["text"]) for c in chunks]
        self.N      = len(self.corpus)
        self.df     = Counter()
        total_len   = 0
        for tokens in self.corpus:
            total_len += len(tokens)
            for t in set(tokens):
                self.df[t] += 1
        self.avgdl = total_len / self.N if self.N else 1

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.corpus:
            return []
        q_tokens = self.tokenize(query)
        scores   = []
        for i, tokens in enumerate(self.corpus):
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                idf = math.log((self.N - self.df[t] + 0.5) /
                               (self.df[t] + 0.5) + 1)
                tf_score = (tf[t] * (self.k1 + 1)) / (
                    tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                score += idf * tf_score
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in scores[:top_k]:
            if score > 0:
                chunk = dict(self.raw[i])
                chunk["bm25_score"]  = round(score, 4)
                chunk["confidence"]  = min(0.95, score / 10)
                results.append(chunk)
        return results


# Per-collection BM25 index cache
_bm25_cache: dict[str, BM25] = {}


def get_bm25(collection_name: str) -> BM25:
    if collection_name not in _bm25_cache:
        _bm25_cache[collection_name] = BM25()
    return _bm25_cache[collection_name]


def update_bm25(collection_name: str, chunks: list[dict]):
    bm25 = get_bm25(collection_name)
    bm25.index(chunks)


# ════════════════════════════════════════════════════════════
# VECTOR SEARCH
# ════════════════════════════════════════════════════════════

def vector_search(collection_name: str, question: str,
                  page_filter: list[int] = None, n: int = 5) -> list[dict]:
    try:
        client = get_chroma()
        col    = client.get_collection(collection_name)
        emb    = get_embedder().encode([question]).tolist()
        kwargs = {
            "query_embeddings": emb,
            "n_results":        min(n, col.count()),
            "include":          ["documents", "metadatas", "distances"]
        }
        if page_filter:
            kwargs["where"] = {"page": {"$in": page_filter}}

        res    = col.query(**kwargs)
        chunks = []
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0]
        ):
            chunks.append({
                "text":       doc,
                "page":       meta.get("page", 0),
                "source":     meta.get("source", ""),
                "upload_order": meta.get("upload_order", 0),
                "doc_date":   meta.get("doc_date", ""),
                "confidence": round(max(0, 1 - dist), 3),
            })
        return chunks
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return []


# ════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL: BM25 + Vector + LLM Page Reasoning
# ════════════════════════════════════════════════════════════

def hybrid_retrieve(collection_name: str, question: str,
                    doc_structure: list[dict], n: int = 6) -> list[dict]:
    """
    1. BM25 keyword search (handles exact financial/legal terms)
    2. Vector semantic search (handles conceptual queries)
    3. LLM page reasoning (decides which section to look in)
    4. Merge and deduplicate, ranked by combined score
    """

    # Step 1 — BM25
    bm25    = get_bm25(collection_name)
    bm25_r  = bm25.search(question, top_k=n) if bm25.corpus else []

    # Step 2 — LLM page reasoning
    page_indices = []
    if doc_structure:
        structure_text = "\n".join(
            [f"Page {d['page']}: {d['summary']}" for d in doc_structure[:40]]
        )
        prompt = f"""Document structure:
{structure_text}

Question: {question}

Which page numbers most likely contain the answer?
Reply ONLY with a JSON array of integers e.g. [3, 7, 12]"""
        resp = llm_call(prompt, max_tokens=64)
        try:
            match = re.search(r'\[[\d,\s]+\]', resp)
            if match:
                page_indices = [int(p) for p in json.loads(match.group())]
        except Exception:
            pass

    # Step 3 — Vector search (with page hint if available)
    vec_r = vector_search(collection_name, question, page_indices or None, n)
    if not vec_r:
        vec_r = vector_search(collection_name, question, None, n)

    # Step 4 — Merge + deduplicate by text
    seen   = set()
    merged = []
    for chunk in bm25_r + vec_r:
        key = chunk["text"][:100]
        if key not in seen:
            seen.add(key)
            # Combined score: average of bm25 and vector confidence
            bm25_conf = chunk.get("bm25_score", 0) / 10
            vec_conf  = chunk.get("confidence", 0)
            chunk["confidence"] = round((bm25_conf + vec_conf) / 2
                                        if bm25_conf > 0 else vec_conf, 3)
            merged.append(chunk)

    merged.sort(key=lambda x: x["confidence"], reverse=True)
    return merged[:n]


# ════════════════════════════════════════════════════════════
# RECENCY DETECTION
# ════════════════════════════════════════════════════════════

def detect_doc_date(text: str) -> str:
    """Extract date from document text. Returns ISO string or empty."""
    patterns = [
        r'\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b',
        r'\b(January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2},?\s+(20\d{2})\b',
        r'\bamended\s+(?:on\s+)?(\w+\s+\d{1,2},?\s+20\d{2})\b',
        r'\beffective\s+(?:date[:\s]+)?(\w+\s+\d{1,2},?\s+20\d{2})\b',
        r'\bversion\s+[\d.]+\s*[,\-]\s*(\w+\s+20\d{2})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def get_doc_recency_order(chunks: list[dict]) -> dict[str, int]:
    """
    Assign recency rank to each source document.
    1 = most recent (highest priority), higher number = older.
    Priority: extracted date > upload order.
    """
    sources     = {}
    for c in chunks:
        src = c.get("source", "unknown")
        if src not in sources:
            sources[src] = {
                "doc_date":    c.get("doc_date", ""),
                "upload_order": c.get("upload_order", 0),
            }

    # Sort by doc_date desc, then upload_order desc
    ranked = sorted(
        sources.items(),
        key=lambda x: (x[1]["doc_date"] or "", x[1]["upload_order"]),
        reverse=True
    )
    return {src: rank + 1 for rank, (src, _) in enumerate(ranked)}


# ════════════════════════════════════════════════════════════
# AUTO DOMAIN DETECTION
# ════════════════════════════════════════════════════════════

DOMAIN_SIGNALS = {
    "legal": ["law", "regulation", "statute", "section", "clause", "pursuant",
              "whereas", "hereby", "jurisdiction", "compliance", "amendment",
              "effective date", "provision", "parties", "agreement", "contract"],
    "financial": ["revenue", "profit", "loss", "ebitda", "q1", "q2", "q3", "q4",
                  "balance sheet", "income statement", "cash flow", "earnings",
                  "fiscal", "quarterly", "annual report", "dividend", "margin"],
    "scientific": ["hypothesis", "methodology", "abstract", "results", "conclusion",
                   "experiment", "data", "analysis", "findings", "study", "research"],
    "general": []
}

def detect_domain(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for domain, signals in DOMAIN_SIGNALS.items():
        scores[domain] = sum(1 for s in signals if s in text_lower)
    scores.pop("general")
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ════════════════════════════════════════════════════════════
# DOC-VS-DOC CONTRADICTION DETECTION
# ════════════════════════════════════════════════════════════

def detect_contradiction(question: str, chunks: list[dict],
                         domain: str) -> dict:
    """
    Check if chunks from different documents contradict each other.
    Returns contradiction info if found.
    """
    # Group chunks by source document
    by_source = {}
    for c in chunks:
        src = c.get("source", "unknown")
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(c["text"])

    if len(by_source) < 2:
        return {}  # Need at least 2 docs to find contradiction

    # Get recency ranking
    recency = get_doc_recency_order(chunks)
    sources_ranked = sorted(by_source.keys(),
                           key=lambda s: recency.get(s, 999))
    newest_src = sources_ranked[0]
    oldest_src = sources_ranked[-1]

    newest_text = " ".join(by_source[newest_src])[:600]
    oldest_text = " ".join(by_source[oldest_src])[:600]

    prompt = f"""You are a document analyst checking for contradictions.

Question asked: {question}

Newest document excerpt:
{newest_text}

Older document excerpt:
{oldest_text}

Do these documents contradict each other on the topic of the question?
If yes, extract the specific conflicting statements.

Respond in this exact JSON format:
{{
  "contradiction_found": true/false,
  "newest_says": "exact claim from newest doc or empty string",
  "older_says": "exact claim from older doc or empty string",
  "topic": "brief topic of contradiction"
}}

Reply ONLY with the JSON."""

    resp = llm_call(prompt, max_tokens=256)
    try:
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        if match:
            result = json.loads(match.group())
            if result.get("contradiction_found"):
                result["newest_source"]  = newest_src
                result["older_source"]   = oldest_src
                result["domain"]         = domain
                result["is_legal_domain"] = domain in LEGAL_DOMAINS
                return result
    except Exception:
        pass
    return {}


# ════════════════════════════════════════════════════════════
# CAUSATION ANALYSIS (legal/regulatory domains only)
# ════════════════════════════════════════════════════════════

def analyze_causation(question: str, contradiction: dict,
                      chunks: list[dict]) -> dict:
    """
    Multi-hop reasoning: search for override/bypass clauses
    that explain why the newer document differs from the older one.
    Only triggered for legal/regulatory domains.
    """
    newer_claim = contradiction.get("newest_says", "")
    older_claim = contradiction.get("older_says", "")
    all_text    = "\n".join([c["text"] for c in chunks])[:2000]

    prompt = f"""You are a legal document analyst.

Original rule: {older_claim}
Updated rule:  {newer_claim}

Full document context:
{all_text}

Search the document context for:
1. Any amendment, override, or bypass clause that explains why the rule changed
2. Any exception or condition that would make the newer rule apply

Respond in this exact JSON format:
{{
  "explanation_found": true/false,
  "explanation": "explanation of why the change occurred, or empty string",
  "override_clause": "the specific clause that enables the override, or empty string",
  "verdict": "NEWER_APPLIES / OLDER_APPLIES / CONTRADITION_UNRESOLVED"
}}

Reply ONLY with the JSON."""

    resp = llm_call(prompt, max_tokens=300)
    try:
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {
        "explanation_found": False,
        "explanation":       "",
        "override_clause":   "",
        "verdict":           "CONTRADICTION_UNRESOLVED"
    }


# ════════════════════════════════════════════════════════════
# CONFIDENCE SCORING
# ════════════════════════════════════════════════════════════

def compute_confidence(chunks: list[dict], answer: str) -> float:
    if not chunks:
        return 0.0
    base = sum(c.get("confidence", 0) for c in chunks) / len(chunks)
    uncertainty = ["i'm not sure", "i cannot find", "not explicitly",
                   "unclear", "insufficient", "no information"]
    penalty = sum(0.08 for p in uncertainty if p in answer.lower())
    return max(0.0, min(1.0, base - penalty))


def format_confidence(score: float) -> dict:
    pct = round(score * 100)
    if score >= 0.5:
        return {"display": f"{pct}%", "type": "percentage", "score": score}
    return {
        "display": (f"Low confidence ({pct}%) — this may not be accurate "
                    "or what you are looking for due to insufficient evidence."),
        "type":    "written",
        "score":   score
    }


# ════════════════════════════════════════════════════════════
# ANSWER GENERATION
# ════════════════════════════════════════════════════════════

def generate_answer(question: str, chunks: list[dict],
                    recency: dict = None) -> str:
    if not chunks:
        return ("I could not find relevant information in your documents. "
                "Please ensure the relevant document has been uploaded.")

    # Sort chunks by recency (newest first)
    if recency:
        chunks = sorted(chunks,
                        key=lambda c: recency.get(c.get("source",""), 999))

    context = "\n\n".join(
        [f"[{c.get('source','Doc')} | Page {c.get('page','')}]\n{c['text']}"
         for c in chunks[:5]]
    )
    system = """You are a precise document analyst.
Answer using ONLY the provided context.
When documents contradict each other, state what each says and which is newer.
Always prefer the most recently dated or versioned document.
Cite exact values, dates, and figures. Never invent information not in the context."""

    prompt = f"""Context from documents:
{context}

Question: {question}

Answer using only the context above. If multiple documents give different answers,
state each value and identify which document is newer."""

    return llm_call(prompt, system=system, max_tokens=512) or \
           "Unable to generate answer. Please check Ollama is running."


# ════════════════════════════════════════════════════════════
# FULL USER DOC PIPELINE
# ════════════════════════════════════════════════════════════

def query_user_docs(collection_name: str, question: str,
                    doc_structure: list[dict],
                    check_contradictions: bool = False) -> dict:
    """
    Full pipeline:
    1. Hybrid retrieve (BM25 + Vector + LLM page reasoning)
    2. Detect domain from retrieved content
    3. Check doc-vs-doc contradictions if multiple docs
    4. Causation analysis for legal domains
    5. Generate answer weighted by recency
    6. Confidence scoring
    """

    # Step 1 — Hybrid retrieval
    chunks = hybrid_retrieve(collection_name, question, doc_structure)

    if not chunks:
        return {
            "answer":         "No relevant content found in your documents.",
            "confidence":     format_confidence(0.0),
            "contradiction":  {},
            "causation":      {},
            "sources":        [],
        }

    # Step 2 — Domain detection
    combined_text = " ".join([c["text"] for c in chunks[:3]])
    domain        = detect_domain(combined_text)

    # Step 3 — Recency ranking
    recency = get_doc_recency_order(chunks)

    # Step 4 — Contradiction detection
    contradiction = {}
    causation     = {}
    if check_contradictions or len(set(c.get("source","") for c in chunks)) > 1:
        contradiction = detect_contradiction(question, chunks, domain)
        if contradiction.get("contradiction_found"):
            # Step 5 — Causation analysis (legal domains only)
            if domain in LEGAL_DOMAINS:
                causation = analyze_causation(question, contradiction, chunks)

    # Step 6 — Generate answer
    answer = generate_answer(question, chunks, recency)

    # Step 7 — Confidence
    conf_score = compute_confidence(chunks, answer)
    confidence = format_confidence(conf_score)

    sources = list({c.get("source","") for c in chunks if c.get("source")})

    return {
        "answer":        answer,
        "confidence":    confidence,
        "contradiction": contradiction,
        "causation":     causation,
        "sources":       sources,
        "domain":        domain,
        "chunks_used":   len(chunks),
    }


# ════════════════════════════════════════════════════════════
# PUBLIC LIBRARY QUERY
# ════════════════════════════════════════════════════════════

def query_public_library(question: str, pub_chunks: list[dict]) -> dict:
    """Simple answer generation from public source chunks."""
    if not pub_chunks:
        # Fall back to Ollama knowledge
        system = """You are a knowledgeable assistant.
Answer accurately from your knowledge.
Be factual and comprehensive."""
        answer    = llm_call(question, system, max_tokens=512)
        conf_raw  = llm_call(
            f"Rate confidence 0-100 for: {question}. Reply with a number only.",
            max_tokens=8
        )
        try:
            score = min(int(''.join(filter(str.isdigit, conf_raw[:4]))), 92) / 100
        except Exception:
            score = 0.72
        note = " (from model knowledge — live sources unavailable)"
        if score >= 0.5:
            conf = {"score": score, "display": f"{round(score*100)}%{note}",
                    "type": "percentage"}
        else:
            conf = {"score": score,
                    "display": f"Low confidence ({round(score*100)}%) — limited information.",
                    "type": "written"}
        return {"answer": answer, "confidence": conf, "sources": []}

    context = "\n\n".join(
        [f"[{c['source']}]\n{c['text']}" for c in pub_chunks[:5]]
    )
    system = """You are a knowledgeable assistant with access to public sources.
Answer based on the provided sources. Cite the source name.
Be accurate and comprehensive."""
    answer  = llm_call(
        f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer citing sources.",
        system, max_tokens=512
    )
    sources = [c["source"] for c in pub_chunks]
    conf    = {"score": 0.78, "display": "78%", "type": "percentage"}
    return {"answer": answer, "confidence": conf, "sources": sources}
