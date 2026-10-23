import base64

from cv_agent.pdf_utils import extract_text, page_count, render_pages
from tests.conftest import make_pdf


def test_extract_text_skips_empty_pages(pdf_bytes: bytes) -> None:
    text = extract_text(pdf_bytes)
    assert "--- Page 1 ---" in text
    assert "Jane Doe" in text
    assert "--- Page 2 ---" not in text
    assert "--- Page 3 ---" in text
    assert "Skills: Python" in text


def test_extract_text_without_text_layer() -> None:
    assert extract_text(make_pdf([""])) == ""


def test_render_pages_returns_png(pdf_bytes: bytes) -> None:
    images = render_pages(pdf_bytes)
    assert len(images) == 3
    assert base64.b64decode(images[0]).startswith(b"\x89PNG")


def test_render_pages_respects_max_pages(pdf_bytes: bytes) -> None:
    assert len(render_pages(pdf_bytes, max_pages=1)) == 1


def test_render_pages_caps_at_page_count() -> None:
    assert len(render_pages(make_pdf(["only"]), max_pages=6)) == 1


def test_page_count(pdf_bytes: bytes) -> None:
    assert page_count(pdf_bytes) == 3
