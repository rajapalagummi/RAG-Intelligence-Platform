"""
PDF Report Generator
====================
Generates clean PDF reports from RAG query results.
Only includes content from the explicit report request — not prior conversation.
"""

import logging
from datetime import datetime
from pathlib import Path
from fpdf import FPDF

logger     = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(30, 30, 60)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "RAG Intelligence Platform - Report", align="C",
                  fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  align="C")


def safe_text(text: str) -> str:
    """Remove non-latin1 characters for FPDF compatibility."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_report(
    question:        str,
    result:          dict,
    public_results:  list[dict] = None,
    report_title:    str = "Analysis Report",
    chart_data:      dict = None,
) -> str:
    """
    Generate a PDF report for a single RAG query result.
    Returns the path to the generated PDF.
    """
    pdf = ReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Title ──────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 60)
    pdf.multi_cell(0, 10, safe_text(report_title), align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Divider ─────────────────────────────────────────────
    pdf.set_draw_color(30, 30, 60)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # ── Query ───────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 8, "Query", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.set_fill_color(245, 245, 250)
    pdf.multi_cell(0, 7, safe_text(question), fill=True)
    pdf.ln(5)

    # ── Answer ──────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 8, "Answer", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    answer = result.get("answer", "No answer generated.")
    pdf.multi_cell(0, 7, safe_text(answer))
    pdf.ln(4)

    # ── Confidence ──────────────────────────────────────────
    conf = result.get("confidence", {})
    if conf:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 30, 60)
        pdf.cell(0, 7, "Confidence Assessment", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)

        score = conf.get("score", 0)
        if score >= 0.5:
            pdf.set_text_color(0, 128, 0)
            pdf.cell(0, 6, f"Confidence: {conf.get('display', '')}",
                     new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_text_color(200, 100, 0)
            pdf.multi_cell(0, 6, safe_text(conf.get("display", "")))
        pdf.set_text_color(40, 40, 40)
        pdf.ln(4)

    # ── Sources ─────────────────────────────────────────────
    sources = result.get("sources", [])
    if sources:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 30, 60)
        pdf.cell(0, 7, "Document Sources", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(60, 60, 60)
        for s in sources:
            pdf.cell(6, 6, chr(149))  # bullet
            pdf.multi_cell(0, 6, safe_text(str(s)[:120]))
        pdf.ln(4)

    # ── Contradictions ──────────────────────────────────────
    contradictions = result.get("contradictions", [])
    if contradictions:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(180, 30, 30)
        pdf.cell(0, 8, "Contradictions Found — Public Sources",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(180, 30, 30)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        for i, c in enumerate(contradictions, 1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(0, 7, f"Contradiction {i}:", new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("Helvetica", "", 9)
            pdf.set_fill_color(255, 245, 245)

            pdf.set_text_color(100, 30, 30)
            pdf.cell(30, 6, "Your document:", new_x="RIGHT", new_y="TOP")
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 6, safe_text(c.get("user_says", "")), fill=True)

            pdf.set_text_color(30, 100, 30)
            pdf.cell(30, 6, "Public source:", new_x="RIGHT", new_y="TOP")
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 6, safe_text(c.get("public_says", "")), fill=True)

            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(128, 128, 128)
            pdf.multi_cell(0, 5, f"Source: {safe_text(c.get('source', ''))}")
            pdf.ln(4)

    # ── Public Results ──────────────────────────────────────
    if public_results:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 60)
        pdf.cell(0, 8, "Related Public Sources", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(30, 30, 60)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)

        for i, r in enumerate(public_results[:5], 1):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 30, 60)
            pdf.multi_cell(0, 7, safe_text(f"{i}. {r.get('title','')}"))

            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 6, safe_text(r.get("text", "")[:300] + "..."))

            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(100, 100, 180)
            pdf.multi_cell(0, 5, safe_text(r.get("source", "")))
            pdf.ln(3)

    # ── Chart note ───────────────────────────────────────────
    if chart_data and chart_data.get("chart_type","none") != "none":
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 60)
        pdf.cell(0, 8, "Chart Data", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        labels = chart_data.get("labels", [])
        values = chart_data.get("values", [])
        for label, value in zip(labels, values):
            pdf.cell(0, 6, safe_text(f"  {label}: {value}"),
                     new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 6,
            safe_text("Note: Open the HTML chart file for the interactive visualisation."))

    # ── Save ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = OUTPUT_DIR / f"report_{timestamp}.pdf"
    pdf.output(str(filename))
    logger.info(f"  ✅ Report saved: {filename}")
    return str(filename)
