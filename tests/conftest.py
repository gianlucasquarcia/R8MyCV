import base64

import pymupdf
import pytest


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf(["Jane Doe - Software Engineer", "", "Skills: Python"])


@pytest.fixture
def png_base64(pdf_bytes: bytes) -> str:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return base64.b64encode(doc[0].get_pixmap().tobytes("png")).decode("ascii")
