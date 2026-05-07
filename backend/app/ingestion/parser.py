from pathlib import Path

import fitz  # PyMuPDF


def parse_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text page-by-page from a PDF.

    Returns a list of dicts: {page_num, content, char_count}.
    Pages with no extractable text (e.g. scanned images) are skipped.
    """
    doc = fitz.open(str(file_path))
    pages: list[dict] = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        cleaned = text.strip()
        if cleaned:
            pages.append(
                {
                    "page_num": page_num,
                    "content": cleaned,
                    "char_count": len(cleaned),
                }
            )

    doc.close()
    return pages


def full_text(pages: list[dict]) -> str:
    """Concatenate all pages into a single string for document-level processing."""
    return "\n\n".join(p["content"] for p in pages)
