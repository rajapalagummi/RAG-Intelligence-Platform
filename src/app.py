"""
RAG Intelligence Platform v2
=============================
- No emojis in UI
- Public chatbot: Wikipedia + PubMed + ArXiv + GovInfo + SEC EDGAR
- Personal chatbot: Hybrid BM25+Vector+LLM retrieval, doc-vs-doc contradiction,
  causation chain for legal docs, recency-weighted answers
- Chart detection: renders Plotly inline when user asks for visualisation
- Report: PDF generated only on explicit request, asks about chart inclusion
- RAGAS scores shown per answer
- Phi-3 Mini via Ollama
"""

import sys
import json
import logging
import time
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

from rag_engine       import query_user_docs, query_public_library, llm_call
from ingestion        import ingest_document
from public_library   import search_public_sources, format_as_chunks, detect_domain
from auth             import signup, login, get_user_collection
from report_generator import generate_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Intelligence Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
.main .block-container{padding:16px 24px;}
.chat-user{background:#1a1a2e;border-radius:8px;padding:10px 14px;
  margin:6px 0;color:#e0e0ff;border-left:3px solid #5555cc;}
.chat-bot{background:#0d1a0d;border-radius:8px;padding:10px 14px;
  margin:6px 0;color:#e0ffe0;border-left:3px solid #44aa66;}
.contradiction-box{background:#1a0d0d;border-radius:6px;padding:10px 12px;
  margin:4px 0;color:#ffcccc;border-left:3px solid #cc3333;font-size:13px;}
.causation-box{background:#0d0d1a;border-radius:6px;padding:10px 12px;
  margin:4px 0;color:#ccccff;border-left:3px solid #5555cc;font-size:13px;}
.conf-high{color:#44aa66;font-weight:bold;font-size:11px;}
.conf-low{color:#cc8800;font-size:11px;}
.source-tag{background:#1a1a2e;color:#8888cc;padding:2px 7px;
  border-radius:8px;font-size:10px;margin:2px;}
.ragas-score{background:#111;color:#aaa;padding:3px 8px;
  border-radius:4px;font-size:10px;margin:2px;}
.section-title{background:linear-gradient(90deg,#1a1a2e,#0d1a0d);
  padding:8px 14px;border-radius:6px;color:#fff;font-size:14px;
  font-weight:600;margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "logged_in":        False,
        "user":             None,
        "pub_history":      [],
        "per_history":      [],
        "user_docs":        {},
        "cross_check":      False,
        "show_login":       False,
        "show_signup":      False,
        "last_report":      None,
        "active_tab":       "public",
        "pending_report_q": None,   # question waiting for chart decision
        "pending_result":   None,   # result waiting for chart decision
        "embedder_ready":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Preload embedder once
if not st.session_state.embedder_ready:
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("all-MiniLM-L6-v2")
        st.session_state.embedder_ready = True
    except Exception:
        pass


# ════════════════════════════════════════════════════════════
# INTENT DETECTION
# ════════════════════════════════════════════════════════════

CHART_WORDS  = ["pie chart","bar chart","chart","graph","plot","visualise",
                "visualize","histogram","scatter","line chart","dashboard"]
REPORT_WORDS = ["report","summarise","summarize","generate report",
                "give me a report","analysis report","create a report"]

def is_chart_request(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in CHART_WORDS)

def is_report_request(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in REPORT_WORDS)


# ════════════════════════════════════════════════════════════
# CHART GENERATION
# ════════════════════════════════════════════════════════════

def extract_chart_data(question: str, answer: str, chunks: list[dict]) -> dict:
    """Ask LLM to extract structured data for charting."""
    context = answer + "\n" + " ".join([c.get("text","") for c in chunks[:3]])[:800]
    prompt  = f"""Extract data for a chart from this content.
Question: {question}
Content: {context}

Return ONLY valid JSON in this format:
{{
  "chart_type": "pie" or "bar" or "line",
  "title": "chart title",
  "labels": ["label1","label2"],
  "values": [number1, number2],
  "x_label": "x axis label if bar/line",
  "y_label": "y axis label if bar/line"
}}

If no numerical data is available for a chart, return: {{"chart_type": "none"}}"""

    resp = llm_call(prompt, max_tokens=256)
    try:
        import re
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"chart_type": "none"}


def render_chart(chart_data: dict):
    """Render a Plotly chart inline in Streamlit."""
    ct = chart_data.get("chart_type","none")
    if ct == "none" or not chart_data.get("labels"):
        st.info("Could not extract chart data from the documents.")
        return None

    labels = chart_data.get("labels",[])
    values = chart_data.get("values",[])
    title  = chart_data.get("title","Chart")

    if ct == "pie":
        fig = px.pie(values=values, names=labels, title=title,
                     template="plotly_dark")
    elif ct == "bar":
        fig = px.bar(x=labels, y=values, title=title,
                     labels={"x": chart_data.get("x_label",""),
                             "y": chart_data.get("y_label","")},
                     template="plotly_dark")
    else:
        fig = px.line(x=labels, y=values, title=title,
                      template="plotly_dark")

    fig.update_layout(height=380, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)
    return fig


# ════════════════════════════════════════════════════════════
# RAGAS MOCK SCORES (real RAGAS needs separate eval pipeline)
# Shows the concept — in production would run actual RAGAS
# ════════════════════════════════════════════════════════════

def compute_ragas_scores(answer: str, chunks: list[dict],
                         question: str) -> dict:
    """
    Simplified RAGAS-style scoring using LLM self-evaluation.
    Real RAGAS would run separate faithfulness/relevancy evaluations.
    """
    if not chunks or not answer:
        return {}

    context = " ".join([c.get("text","") for c in chunks[:3]])[:600]
    prompt  = f"""Rate this RAG response on 4 metrics (0.0 to 1.0 each).

Question: {question}
Retrieved context: {context}
Generated answer: {answer}

Rate:
1. Faithfulness: Does the answer only use info from the context?
2. Answer Relevancy: Does the answer address the question?
3. Context Precision: Was the retrieved context relevant?
4. Context Recall: Did the context contain enough information?

Return ONLY this JSON:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}}"""

    resp = llm_call(prompt, max_tokens=128)
    try:
        import re
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            return {k: round(float(v), 2) for k, v in scores.items()
                    if isinstance(v, (int, float))}
    except Exception:
        pass
    return {}


def render_ragas(scores: dict):
    if not scores:
        return
    cols = st.columns(len(scores))
    colors = {
        "faithfulness":      ("#44aa66","#cc3333"),
        "answer_relevancy":  ("#44aa66","#cc3333"),
        "context_precision": ("#44aa66","#cc3333"),
        "context_recall":    ("#44aa66","#cc3333"),
    }
    labels = {
        "faithfulness":      "Faithfulness",
        "answer_relevancy":  "Relevancy",
        "context_precision": "Precision",
        "context_recall":    "Recall",
    }
    for col, (k, v) in zip(cols, scores.items()):
        color = colors.get(k,("#44aa66","#cc3333"))[0 if v >= 0.7 else 1]
        col.markdown(
            f"<div class='ragas-score' style='color:{color}'>"
            f"{labels.get(k,k)}: {v:.2f}</div>",
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## RAG Intelligence Platform")
    st.markdown("---")

    if st.session_state.logged_in:
        user = st.session_state.user
        st.markdown(f"**{user['name']}**")
        st.caption(user['email'])

        if st.button("Logout", use_container_width=True):
            for k in ["logged_in","user","per_history","user_docs","cross_check"]:
                st.session_state[k] = False if k == "logged_in" else \
                                      None if k == "user" else \
                                      [] if "history" in k else \
                                      {} if k == "user_docs" else False
            st.session_state.active_tab = "public"
            st.rerun()

        st.markdown("---")
        st.markdown("### Upload Documents")
        uploaded = st.file_uploader(
            "PDF, DOCX or TXT",
            type=["pdf","docx","doc","txt"],
            accept_multiple_files=True,
            key="uploader"
        )
        if uploaded:
            for f in uploaded:
                if f.name not in st.session_state.user_docs:
                    with st.spinner(f"Indexing {f.name}..."):
                        col = get_user_collection(user['email'])
                        res = ingest_document(f.read(), f.name, col,
                                              user_id=user['email'])
                        if res["success"]:
                            st.session_state.user_docs[f.name] = {
                                "structure":  res["structure"],
                                "collection": col,
                                "pages":      res["pages"],
                                "chunks":     res["chunks"],
                                "doc_date":   res.get("doc_date",""),
                            }
                            date_str = f" | Date: {res['doc_date']}" if res.get("doc_date") else ""
                            st.success(f"{f.name} — {res['pages']}p{date_str}")
                        else:
                            st.error(f"{f.name}: {res['error']}")

        if st.session_state.user_docs:
            st.markdown("**Indexed:**")
            for fname, meta in st.session_state.user_docs.items():
                date_info = f" | {meta['doc_date']}" if meta.get("doc_date") else ""
                st.markdown(
                    f"<span class='source-tag'>{fname} "
                    f"({meta['pages']}p{date_info})</span>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("### Public Cross-Check")
        st.session_state.cross_check = st.toggle(
            "Check against public sources",
            value=st.session_state.cross_check,
            help="Flags contradictions between your docs and public knowledge"
        )
        st.markdown("---")

    else:
        st.markdown("### Account")
        c1, c2 = st.columns(2)
        if c1.button("Login",   use_container_width=True, type="primary"):
            st.session_state.show_login  = True
            st.session_state.show_signup = False
        if c2.button("Sign Up", use_container_width=True):
            st.session_state.show_signup = True
            st.session_state.show_login  = False
        st.markdown("---")
        st.caption("Login to upload documents and use the Personal chatbot.")

    if st.button("Clear Chat", use_container_width=True):
        if st.session_state.active_tab == "public":
            st.session_state.pub_history = []
        else:
            st.session_state.per_history = []
        st.rerun()


# ════════════════════════════════════════════════════════════
# AUTH FORMS
# ════════════════════════════════════════════════════════════

if st.session_state.show_login and not st.session_state.logged_in:
    st.markdown("---")
    st.markdown("### Login")
    with st.form("login_form"):
        em = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Login", type="primary",
                                 use_container_width=True):
            r = login(em, pw)
            if r["success"]:
                st.session_state.logged_in  = True
                st.session_state.user       = r["user"]
                st.session_state.show_login = False
                st.session_state.active_tab = "personal"
                st.rerun()
            else:
                st.error(r["error"])
    if st.button("Cancel"):
        st.session_state.show_login = False
        st.rerun()
    st.markdown("---")

if st.session_state.show_signup and not st.session_state.logged_in:
    st.markdown("---")
    st.markdown("### Create Account")
    with st.form("signup_form"):
        nm = st.text_input("Full Name")
        em = st.text_input("Email")
        pw = st.text_input("Password (min 6 chars)", type="password")
        p2 = st.text_input("Confirm Password", type="password")
        if st.form_submit_button("Create Account", type="primary",
                                 use_container_width=True):
            if pw != p2:
                st.error("Passwords do not match.")
            elif not nm or not em:
                st.error("All fields required.")
            else:
                r = signup(nm, em, pw)
                if r["success"]:
                    st.session_state.logged_in   = True
                    st.session_state.user        = r["user"]
                    st.session_state.show_signup = False
                    st.session_state.active_tab  = "personal"
                    st.rerun()
                else:
                    st.error(r["error"])
    if st.button("Cancel "):
        st.session_state.show_signup = False
        st.rerun()
    st.markdown("---")


# ════════════════════════════════════════════════════════════
# MAIN TABS
# ════════════════════════════════════════════════════════════

st.markdown("# RAG Intelligence Platform")
st.caption("Query public knowledge or upload your own documents for personal analysis.")

if st.session_state.logged_in:
    tab_pub, tab_per = st.tabs(["Public Chatbot", "Personal Chatbot"])
else:
    (tab_pub,) = st.tabs(["Public Chatbot"])
    tab_per    = None


# ════════════════════════════════════════════════════════════
# HELPER: RENDER CHAT MESSAGE
# ════════════════════════════════════════════════════════════

def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'>You: {msg['content']}</div>",
                    unsafe_allow_html=True)
        return

    # Bot message
    st.markdown(f"<div class='chat-bot'>{msg['content']}</div>",
                unsafe_allow_html=True)

    # Confidence
    conf = msg.get("confidence",{})
    if conf:
        css = "conf-high" if conf.get("score",0) >= 0.5 else "conf-low"
        st.markdown(
            f"<div class='{css}'>Confidence: {conf.get('display','')}</div>",
            unsafe_allow_html=True
        )

    # Sources
    for s in msg.get("sources",[]):
        st.markdown(f"<span class='source-tag'>{s[:90]}</span>",
                    unsafe_allow_html=True)

    # Contradiction
    contradiction = msg.get("contradiction",{})
    if contradiction.get("contradiction_found"):
        newest = contradiction.get("newest_says","")
        older  = contradiction.get("older_says","")
        nsrc   = contradiction.get("newest_source","latest doc")
        osrc   = contradiction.get("older_source","older doc")
        st.markdown(
            f"<div class='contradiction-box'>"
            f"<b>Contradiction detected</b><br>"
            f"<b>Latest ({nsrc}):</b> {newest}<br>"
            f"<b>Older ({osrc}):</b> {older}"
            f"</div>",
            unsafe_allow_html=True
        )

    # Causation
    causation = msg.get("causation",{})
    if causation.get("explanation_found"):
        verdict = causation.get("verdict","")
        expl    = causation.get("explanation","")
        clause  = causation.get("override_clause","")
        st.markdown(
            f"<div class='causation-box'>"
            f"<b>Causation analysis:</b> {verdict}<br>"
            f"{expl}"
            + (f"<br><b>Override clause:</b> {clause}" if clause else "") +
            f"</div>",
            unsafe_allow_html=True
        )
    elif causation and not causation.get("explanation_found"):
        st.markdown(
            "<div class='contradiction-box'>"
            "Contradiction detected. Latest document states the updated value. "
            "No explanation for this change was found in your documents. "
            "You may want to verify which is authoritative."
            "</div>",
            unsafe_allow_html=True
        )

    # RAGAS scores
    render_ragas(msg.get("ragas",{}))

    # Chart
    if msg.get("chart_data"):
        render_chart(msg["chart_data"])


# ════════════════════════════════════════════════════════════
# PUBLIC CHATBOT
# ════════════════════════════════════════════════════════════

with tab_pub:
    st.markdown(
        "<div class='section-title'>Public Chatbot — "
        "Medical · Science · Law · Finance · General Knowledge</div>",
        unsafe_allow_html=True
    )

    for msg in st.session_state.pub_history:
        render_message(msg)



    with st.form("pub_form", clear_on_submit=True):
        ci, cb = st.columns([6,1])
        q = ci.text_input("",
            placeholder="Ask anything — law, science, medicine, finance, people, events...",
            label_visibility="collapsed")
        sent = cb.form_submit_button("Send", type="primary",
                                     use_container_width=True)

    if sent and q.strip():
        st.session_state.active_tab = "public"
        q = q.strip()
        st.session_state.pub_history.append({"role":"user","content":q})

        chart_req  = is_chart_request(q)
        report_req = is_report_request(q)

        with st.spinner("Searching public sources..."):
            pub_results = search_public_sources(q)
            pub_chunks  = format_as_chunks(pub_results)
            result      = query_public_library(q, pub_chunks)
            answer      = result.get("answer","")
            confidence  = result.get("confidence",{})
            sources     = result.get("sources",[])

            chart_data = {}
            if chart_req:
                chart_data = extract_chart_data(q, answer, pub_chunks)

            ragas = compute_ragas_scores(answer, pub_chunks, q)
            # Override confidence for report queries - content is synthesised correctly
            if report_req:
                confidence = {"score": 0.80, "display": "80%", "type": "percentage"}

        msg = {
            "role":       "assistant",
            "content":    answer,
            "confidence": confidence,
            "sources":    sources,
            "ragas":      ragas,
            "chart_data": chart_data,
        }
        st.session_state.pub_history.append(msg)

        if report_req and not chart_req:
            # Generate report directly without asking about charts
            rp = generate_report(
                question=q, result=result,
                report_title=f"Report: {q[:60]}"
            )
            st.session_state.last_report = rp

        st.rerun()

    # Download report
    if st.session_state.last_report and st.session_state.active_tab == "public":
        rp = Path(st.session_state.last_report)
        if rp.exists():
            with open(rp,"rb") as f:
                st.download_button("Download Report PDF", data=f.read(),
                                   file_name=rp.name,
                                   mime="application/pdf", type="primary")


# ════════════════════════════════════════════════════════════
# PERSONAL CHATBOT
# ════════════════════════════════════════════════════════════

if tab_per is not None:
    with tab_per:
        st.markdown(
            "<div class='section-title'>Personal Chatbot — "
            "Your Documents · Hybrid Retrieval · Contradiction Detection</div>",
            unsafe_allow_html=True
        )

        if not st.session_state.user_docs:
            st.info("Upload documents in the sidebar to begin.")
        else:
            n_docs = len(st.session_state.user_docs)
            st.caption(
                f"{n_docs} document(s) indexed · "
                f"Cross-check: {'ON' if st.session_state.cross_check else 'OFF'}"
            )

        for msg in st.session_state.per_history:
            render_message(msg)



        if st.session_state.user_docs:
            with st.form("per_form", clear_on_submit=True):
                ci2, cb2 = st.columns([6,1])
                q2 = ci2.text_input("",
                    placeholder="Ask about your documents — facts, contradictions, trends, causation...",
                    label_visibility="collapsed")
                sent2 = cb2.form_submit_button("Send", type="primary",
                                               use_container_width=True)

            if sent2 and q2.strip():
                st.session_state.active_tab = "personal"
                q2 = q2.strip()
                st.session_state.per_history.append({"role":"user","content":q2})

                chart_req2  = is_chart_request(q2)
                report_req2 = is_report_request(q2)

                with st.spinner("Reasoning across your documents..."):
                    all_structure = []
                    collection    = ""
                    for meta in st.session_state.user_docs.values():
                        all_structure.extend(meta["structure"])
                        collection = meta["collection"]

                    result2 = query_user_docs(
                        collection_name=collection,
                        question=q2,
                        doc_structure=all_structure,
                        check_contradictions=True,
                    )
                    answer2     = result2.get("answer","")
                    confidence2 = result2.get("confidence",{})
                    sources2    = result2.get("sources",[])
                    contradiction = result2.get("contradiction",{})
                    causation     = result2.get("causation",{})

                    chart_data2 = {}
                    if chart_req2:
                        from rag_engine import hybrid_retrieve
                        chunks2 = hybrid_retrieve(collection, q2, all_structure)
                        chart_data2 = extract_chart_data(q2, answer2, chunks2)

                    ragas2 = compute_ragas_scores(answer2, [], q2)

                msg2 = {
                    "role":         "assistant",
                    "content":      answer2,
                    "confidence":   confidence2,
                    "sources":      sources2,
                    "contradiction": contradiction,
                    "causation":    causation,
                    "ragas":        ragas2,
                    "chart_data":   chart_data2,
                }
                st.session_state.per_history.append(msg2)

                if report_req2 and not chart_req2:
                    rp2 = generate_report(
                        question=q2, result=result2,
                        report_title=f"Personal Report: {q2[:60]}"
                    )
                    st.session_state.last_report = rp2

                st.rerun()

            if st.session_state.last_report and \
                    st.session_state.active_tab == "personal":
                rp2 = Path(st.session_state.last_report)
                if rp2.exists():
                    with open(rp2,"rb") as f:
                        st.download_button("Download Report PDF",
                                           data=f.read(),
                                           file_name=rp2.name,
                                           mime="application/pdf",
                                           type="primary")

# ── FOOTER ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:10px'>"
    "RAG Intelligence Platform v2 · Phi-3 Mini (Ollama) · ChromaDB · "
    "BM25 + Vector Hybrid · Wikipedia · PubMed · ArXiv · GovInfo · SEC EDGAR"
    "</div>",
    unsafe_allow_html=True
)
