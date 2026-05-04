"""
Public Library v2
=================
Domains: Medical (PubMed abstracts), Science (ArXiv),
         Law (GovInfo US Federal), Finance (SEC EDGAR),
         General/News (Wikipedia — catches Lewis Hamilton, Kim K, etc.)
All calls run in parallel with 3s timeout each.
Domain auto-detected from question keywords.
"""

import logging
import requests
import xml.etree.ElementTree as ET
import concurrent.futures
from typing import Optional

logger  = logging.getLogger(__name__)
HEADERS = {"User-Agent": "RAG-Platform-Demo/2.0 (portfolio)"}

DOMAIN_KEYWORDS = {
    "medical":   ["medical","health","disease","treatment","drug","clinical",
                  "patient","symptom","diagnosis","cancer","diabetes","surgery",
                  "hospital","physician","therapy","virus","vaccine","anatomy"],
    "science":   ["physics","chemistry","biology","research","experiment",
                  "quantum","molecule","protein","dna","climate","astronomy",
                  "mathematics","algorithm","neural","gravitational","particle"],
    "law":       ["law","legal","court","ruling","verdict","statute","regulation",
                  "constitution","amendment","federal","supreme","judge","attorney",
                  "plaintiff","defendant","jurisdiction","tort","contract"],
    "finance":   ["stock","revenue","profit","earnings","sec","filing","10-k",
                  "quarterly","balance sheet","cash flow","dividend","market cap",
                  "fiscal","ebitda","shares","equity","debt","financial"],
    "general":   []
}

def detect_domain(question: str) -> str:
    q = question.lower()
    scores = {d: sum(1 for kw in kws if kw in q)
              for d, kws in DOMAIN_KEYWORDS.items() if d != "general"}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ════════════════════════════════════════════════════════════
# WIKIPEDIA — general catch-all
# Handles: people, events, pop culture, history, sports
# ════════════════════════════════════════════════════════════

def search_wikipedia(query: str, limit: int = 2) -> list[dict]:
    results = []
    try:
        # Search for pages
        search_r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action":"query","list":"search","srsearch":query,
                    "srlimit":limit,"format":"json"},
            headers=HEADERS, timeout=3
        )
        pages = search_r.json().get("query",{}).get("search",[])

        for p in pages[:limit]:
            title   = p.get("title","")
            page_id = p.get("pageid","")
            # Fetch extract
            ext_r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action":"query","pageids":page_id,
                        "prop":"extracts","exintro":True,
                        "explaintext":True,"format":"json"},
                headers=HEADERS, timeout=3
            )
            extract = (ext_r.json()
                       .get("query",{})
                       .get("pages",{})
                       .get(str(page_id),{})
                       .get("extract",""))
            if extract:
                results.append({
                    "title":  title,
                    "source": f"Wikipedia — https://en.wikipedia.org/wiki/{title.replace(' ','_')}",
                    "text":   extract[:800],
                    "domain": "general",
                })
    except Exception as e:
        logger.warning(f"Wikipedia error: {e}")
    return results


# ════════════════════════════════════════════════════════════
# PUBMED — medical abstracts
# ════════════════════════════════════════════════════════════

def search_pubmed(query: str, limit: int = 2) -> list[dict]:
    results = []
    try:
        sr = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db":"pubmed","term":query,"retmax":limit,"retmode":"json"},
            headers=HEADERS, timeout=3
        )
        ids = sr.json().get("esearchresult",{}).get("idlist",[])
        if not ids:
            return results
        fr = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db":"pubmed","id":",".join(ids),"retmode":"xml","rettype":"abstract"},
            headers=HEADERS, timeout=4
        )
        root = ET.fromstring(fr.text)
        for art in root.findall(".//PubmedArticle"):
            title_el = art.find(".//ArticleTitle")
            abs_el   = art.find(".//AbstractText")
            pmid_el  = art.find(".//PMID")
            t = title_el.text if title_el is not None else ""
            a = abs_el.text   if abs_el   is not None else ""
            p = pmid_el.text  if pmid_el  is not None else ""
            if t and a:
                results.append({
                    "title":  t,
                    "source": f"PubMed PMID:{p} — https://pubmed.ncbi.nlm.nih.gov/{p}",
                    "text":   f"{t}. {a}",
                    "domain": "medical",
                })
    except Exception as e:
        logger.warning(f"PubMed error: {e}")
    return results


# ════════════════════════════════════════════════════════════
# ARXIV — science preprints
# ════════════════════════════════════════════════════════════

def search_arxiv(query: str, limit: int = 2) -> list[dict]:
    results = []
    try:
        r    = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query":f"all:{query}","max_results":limit},
            headers=HEADERS, timeout=3
        )
        root = ET.fromstring(r.text)
        ns   = {"atom":"http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title   = entry.find("atom:title",   ns)
            summary = entry.find("atom:summary", ns)
            link    = entry.find("atom:id",      ns)
            if title is not None and summary is not None:
                results.append({
                    "title":  title.text.strip(),
                    "source": f"ArXiv — {link.text.strip() if link is not None else 'arxiv.org'}",
                    "text":   f"{title.text.strip()}. {summary.text.strip()[:500]}",
                    "domain": "science",
                })
    except Exception as e:
        logger.warning(f"ArXiv error: {e}")
    return results


# ════════════════════════════════════════════════════════════
# GOVINFO — US Federal Law
# ════════════════════════════════════════════════════════════

def search_us_law(query: str, limit: int = 2) -> list[dict]:
    results = []
    try:
        r = requests.get(
            "https://api.govinfo.gov/search",
            params={"query":query,"pageSize":limit,"offsetMark":"*",
                    "collection":"USCODE,CFR","resultLevel":"default"},
            headers=HEADERS, timeout=3
        )
        for item in r.json().get("results",[])[:limit]:
            title   = item.get("title","")
            snippet = item.get("granuleText", item.get("packageId",""))[:400]
            url     = item.get("detailsLink","https://www.govinfo.gov")
            if title:
                results.append({
                    "title":  title,
                    "source": f"GovInfo US Code — {url}",
                    "text":   f"{title}. {snippet}",
                    "domain": "law",
                })
    except Exception as e:
        logger.warning(f"GovInfo error: {e}")
    return results


# ════════════════════════════════════════════════════════════
# SEC EDGAR — Finance
# ════════════════════════════════════════════════════════════

def search_sec_edgar(query: str, limit: int = 2) -> list[dict]:
    results = []
    try:
        r = requests.get(
            "https://efts.sec.gov/LATEST/search-index?q=%22" +
            query.replace(" ","%20") + "%22&dateRange=custom"
            "&startdt=2020-01-01&forms=10-K,10-Q",
            headers=HEADERS, timeout=3
        )
        hits = r.json().get("hits",{}).get("hits",[])
        for h in hits[:limit]:
            src   = h.get("_source",{})
            title = src.get("display_names",[""])[0] or src.get("file_date","")
            text  = src.get("period_of_report","") + " " + src.get("form_type","")
            url   = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            if title:
                results.append({
                    "title":  title,
                    "source": f"SEC EDGAR — {url}",
                    "text":   f"{title}. {text}",
                    "domain": "finance",
                })
    except Exception as e:
        logger.warning(f"SEC EDGAR error: {e}")
    return results


# ════════════════════════════════════════════════════════════
# OPENLIBRARY — supplementary
# ════════════════════════════════════════════════════════════

def search_openlibrary(query: str, limit: int = 1) -> list[dict]:
    results = []
    try:
        r = requests.get(
            "https://openlibrary.org/search.json",
            params={"q":query,"limit":limit,
                    "fields":"key,title,author_name,first_sentence"},
            headers=HEADERS, timeout=3
        )
        for doc in r.json().get("docs",[])[:limit]:
            title  = doc.get("title","")
            fs     = doc.get("first_sentence",{})
            first  = fs.get("value","") if isinstance(fs,dict) else str(fs)
            key    = doc.get("key","")
            if title and first:
                results.append({
                    "title":  title,
                    "source": f"OpenLibrary — https://openlibrary.org{key}",
                    "text":   f"{title}. {first}",
                    "domain": "general",
                })
    except Exception as e:
        logger.warning(f"OpenLibrary error: {e}")
    return results


# ════════════════════════════════════════════════════════════
# UNIFIED PARALLEL SEARCH
# ════════════════════════════════════════════════════════════

def search_public_sources(question: str, limit: int = 2) -> list[dict]:
    """
    Auto-detect domain, run relevant sources in parallel (3s timeout each).
    Wikipedia always included as catch-all for general knowledge queries.
    """
    domain     = detect_domain(question)
    search_fns = []

    if domain == "medical":
        search_fns = [(search_pubmed, (question, limit)),
                      (search_arxiv,  (question, 1)),
                      (search_wikipedia, (question, 1))]
    elif domain == "science":
        search_fns = [(search_arxiv,     (question, limit)),
                      (search_pubmed,    (question, 1)),
                      (search_wikipedia, (question, 1))]
    elif domain == "law":
        search_fns = [(search_us_law,    (question, limit)),
                      (search_wikipedia, (question, 1))]
    elif domain == "finance":
        search_fns = [(search_sec_edgar, (question, limit)),
                      (search_wikipedia, (question, 1))]
    else:
        # General — Wikipedia primary, OpenLibrary secondary
        search_fns = [(search_wikipedia,  (question, limit)),
                      (search_openlibrary,(question, 1))]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fn, *args): fn.__name__
                   for fn, args in search_fns}
        for future in concurrent.futures.as_completed(futures, timeout=5):
            try:
                results.extend(future.result())
            except Exception as e:
                logger.warning(f"Source {futures[future]} failed: {e}")

    return results[:6]


def format_as_chunks(results: list[dict]) -> list[dict]:
    return [{"text": r["text"], "page": 1,
             "source": r["source"], "confidence": 0.72}
            for r in results]
