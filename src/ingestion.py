"""
Document Ingestion v2
=====================
Extracts text from PDF/DOCX/TXT
Chunks with overlap
Embeds and stores in ChromaDB with recency metadata
Updates BM25 index on every ingest
Extracts document date from text for recency ordering
"""

import io
import uuid
import logging
from pathlib import Path
from typing import Optional

import fitz
import chromadb
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

from rag_engine import (get_embedder, get_chroma, update_bm25,
                        detect_doc_date, BM25)

logger     = logging.getLogger(__name__)
CHUNK_SIZE = 500
OVERLAP    = 50

# Track upload order per collection
_upload_counters: dict[str, int] = {}


# ════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ════════════════════════════════════════════════════════════

def extract_pdf(data: bytes) -> list[dict]:
    pages = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({"page": i+1, "text": text})
    except Exception as e:
        logger.error(f"PDF error: {e}")
    return pages


def extract_docx(data: bytes) -> list[dict]:
    pages, page_num, chunk = [], 1, []
    try:
        doc   = DocxDocument(io.BytesIO(data))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for i, para in enumerate(paras):
            chunk.append(para)
            if len(chunk) >= 10:
                pages.append({"page": page_num, "text": "\n".join(chunk)})
                page_num += 1
                chunk     = []
        if chunk:
            pages.append({"page": page_num, "text": "\n".join(chunk)})
    except Exception as e:
        logger.error(f"DOCX error: {e}")
    return pages


def extract_txt(data: bytes) -> list[dict]:
    text = data.decode("utf-8", errors="ignore")
    return [{"page": i//1000+1, "text": text[i:i+1000]}
            for i in range(0, len(text), 1000) if text[i:i+1000].strip()]


def extract_pages(data: bytes, filename: str) -> list[dict]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":              return extract_pdf(data)
    elif ext in [".docx",".doc"]: return extract_docx(data)
    else:                          return extract_txt(data)


# ════════════════════════════════════════════════════════════
# CHUNKING
# ════════════════════════════════════════════════════════════

def chunk_page(text: str, page: int, source: str) -> list[dict]:
    words  = text.split()
    chunks = []
    step   = CHUNK_SIZE - OVERLAP
    for i in range(0, len(words), step):
        w = words[i:i+CHUNK_SIZE]
        if w:
            chunks.append({"text": " ".join(w), "page": page, "source": source})
    return chunks


def build_page_structure(pages: list[dict]) -> list[dict]:
    return [{"page": p["page"], "summary": p["text"][:150].replace("\n"," ")}
            for p in pages]


# ════════════════════════════════════════════════════════════
# INGEST
# ════════════════════════════════════════════════════════════

def ingest_document(file_bytes: bytes, filename: str,
                    collection_name: str,
                    user_id: Optional[str] = None) -> dict:
    """
    Full ingestion:
    1. Extract pages
    2. Extract doc date for recency ordering
    3. Chunk pages
    4. Embed + store in ChromaDB with metadata
    5. Rebuild BM25 index for this collection
    """
    pages = extract_pages(file_bytes, filename)
    if not pages:
        return {"success": False, "error": "No text extracted", "structure": []}

    # Extract document date from first page text
    doc_date = detect_doc_date(pages[0]["text"]) if pages else ""

    # Upload order counter
    order = _upload_counters.get(collection_name, 0) + 1
    _upload_counters[collection_name] = order

    # Chunk all pages
    all_chunks = []
    for p in pages:
        all_chunks.extend(chunk_page(p["text"], p["page"], filename))

    if not all_chunks:
        return {"success": False, "error": "No chunks", "structure": []}

    # Embed
    embedder   = get_embedder()
    texts      = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    # Store in ChromaDB
    try:
        client = get_chroma()
        try:
            col = client.get_collection(collection_name)
        except Exception:
            col = client.create_collection(collection_name,
                                           metadata={"hnsw:space": "cosine"})

        metadatas = [{
            "page":         c["page"],
            "source":       c["source"],
            "user_id":      user_id or "public",
            "doc_date":     doc_date,
            "upload_order": order,
        } for c in all_chunks]

        col.add(
            ids=[str(uuid.uuid4()) for _ in all_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"  Ingested {len(all_chunks)} chunks for {filename}")
    except Exception as e:
        return {"success": False, "error": str(e), "structure": []}

    # Rebuild BM25 index — fetch ALL chunks for this collection
    try:
        client = get_chroma()
        col    = client.get_collection(collection_name)
        count  = col.count()
        if count > 0:
            all_stored = col.get(
                include=["documents","metadatas"],
                limit=min(count, 50000)
            )
            bm25_chunks = [
                {"text": d, "page": m.get("page",0),
                 "source": m.get("source",""),
                 "upload_order": m.get("upload_order", 0),
                 "doc_date": m.get("doc_date","")}
                for d, m in zip(all_stored["documents"],
                                all_stored["metadatas"])
            ]
            update_bm25(collection_name, bm25_chunks)
    except Exception as e:
        logger.warning(f"BM25 update failed: {e}")

    return {
        "success":   True,
        "filename":  filename,
        "pages":     len(pages),
        "chunks":    len(all_chunks),
        "doc_date":  doc_date,
        "structure": build_page_structure(pages),
    }


# ════════════════════════════════════════════════════════════
# PUBLIC LIBRARY INGEST
# ════════════════════════════════════════════════════════════

def ingest_public_text(text: str, source_id: str, source_url: str,
                       domain: str, title: str = "") -> bool:
    words  = text.split()
    chunks = [" ".join(words[i:i+CHUNK_SIZE])
              for i in range(0, len(words), CHUNK_SIZE-OVERLAP)
              if words[i:i+CHUNK_SIZE]]
    if not chunks:
        return False

    try:
        embedder   = get_embedder()
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
        client     = get_chroma()

        for cname in [f"public_{domain}", "public_general"]:
            try:
                col = client.get_collection(cname)
            except Exception:
                col = client.create_collection(cname,
                                               metadata={"hnsw:space":"cosine"})
            prefix = source_id if cname == f"public_{domain}" else f"gen_{source_id}"
            col.add(
                ids=[f"{prefix}_{i}" for i in range(len(chunks))],
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"source": source_url, "title": title,
                             "domain": domain, "page": i+1,
                             "user_id": "public"}
                           for i in range(len(chunks))]
            )
        return True
    except Exception as e:
        logger.error(f"Public ingest error: {e}")
        return False
