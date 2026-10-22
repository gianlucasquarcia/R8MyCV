"""PDF helpers: text extraction and page rendering for the vision model."""

import base64

import pymupdf


def extract_text(pdf_bytes: bytes) -> str:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text("text").strip() for page in doc]
    return "\n\n".join(
        f"--- Page {i} ---\n{text}" for i, text in enumerate(pages, start=1) if text
    )


def render_pages(pdf_bytes: bytes, max_pages: int = 3, zoom: float = 1.5) -> list[str]:
    """Render the first `max_pages` pages to base64-encoded PNGs."""
    images: list[str] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc.pages(0, min(max_pages, doc.page_count)):
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            images.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
    return images


def page_count(pdf_bytes: bytes) -> int:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count
