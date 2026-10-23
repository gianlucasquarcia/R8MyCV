import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

graph_module = importlib.import_module("cv_agent.graph")
APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

REPORT = {
    "overall_score": 82,
    "summary": "Strong CV",
    "hints": [
        {
            "category": "Impact",
            "priority": "High",
            "issue": "No metrics",
            "suggestion": "Quantify results",
            "example": "Cut latency by 40%",
        },
        {
            "category": "Layout",
            "priority": "Low",
            "issue": "Margins",
            "suggestion": "Widen margins",
        },
    ],
    "missing_keywords": ["Kubernetes", "Terraform"],
}


def full_result(png_base64: str) -> dict:
    return {
        "report": REPORT,
        "layout_review": {"score": 7, "strengths": ["Clean"], "issues": []},
        "content_review": {
            "score": 6,
            "candidate_name": "Jane Doe",
            "current_role": "Engineer",
            "strengths": ["Clear"],
            "weaknesses": ["Short"],
            "detected_sections": ["Experience"],
            "missing_sections": [],
        },
        "page_images": [png_base64],
        "cv_text": "Jane Doe CV",
    }


class FakeGraph:
    def __init__(self, updates=None, error: Exception | None = None) -> None:
        self.updates = updates or []
        self.error = error
        self.calls: list = []

    def stream(self, inputs: dict, context, stream_mode: str):
        self.calls.append((inputs, context, stream_mode))
        if self.error:
            raise self.error
        yield from self.updates


def make_app(monkeypatch: pytest.MonkeyPatch, graph: FakeGraph, uploaded=True):
    monkeypatch.setattr(graph_module, "graph", graph)
    file = SimpleNamespace(name="cv.pdf", getvalue=lambda: b"%PDF-fake")
    monkeypatch.setattr("streamlit.file_uploader", lambda *a, **k: file if uploaded else None)
    return AppTest.from_file(str(APP_PATH), default_timeout=30)


def updates_for(result: dict) -> list[dict]:
    return [{"extract": None}, {"analyze_layout": result}, {"unknown_node": {}}]


def test_initial_render_has_no_results(monkeypatch: pytest.MonkeyPatch) -> None:
    at = make_app(monkeypatch, FakeGraph(), uploaded=False).run()
    assert not at.exception
    assert at.title[0].value == "📄 CV Analyzer"
    assert not at.button
    assert "result" not in at.session_state


def test_analyze_success_renders_report(
    monkeypatch: pytest.MonkeyPatch, png_base64: str
) -> None:
    graph = FakeGraph(updates_for(full_result(png_base64)))
    at = make_app(monkeypatch, graph).run()
    at.selectbox[0].select("English")
    at.text_area[0].input("Kubernetes role")
    at.button[0].click().run()

    assert not at.exception
    inputs, context, mode = graph.calls[0]
    assert inputs["language"] == "English"
    assert inputs["job_description"] == "Kubernetes role"
    assert mode == "updates"
    assert context.max_pages == 3
    assert at.session_state["file_name"] == "cv.pdf"
    assert [m.value for m in at.metric] == ["82/100", "6/10", "7/10"]
    assert any("Missing keywords" in s.value for s in at.subheader)


def test_same_as_cv_language_sends_empty_language(
    monkeypatch: pytest.MonkeyPatch, png_base64: str
) -> None:
    graph = FakeGraph(updates_for(full_result(png_base64)))
    at = make_app(monkeypatch, graph).run()
    at.button[0].click().run()
    assert graph.calls[0][0]["language"] == ""


def test_analyze_failure_shows_error(monkeypatch: pytest.MonkeyPatch) -> None:
    at = make_app(monkeypatch, FakeGraph(error=ConnectionError("no ollama"))).run()
    at.button[0].click().run()
    assert not at.exception
    assert "ConnectionError: no ollama" in at.error[0].value
    assert "Ollama is running" in at.info[0].value
    assert "result" not in at.session_state


def test_saved_result_without_optional_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    at = make_app(monkeypatch, FakeGraph(), uploaded=False)
    at.session_state["result"] = {
        "report": {
            "overall_score": 10,
            "summary": "Thin",
            "hints": [
                {
                    "category": "Layout",
                    "priority": "Medium",
                    "issue": "Spacing",
                    "suggestion": "Add space",
                }
            ],
            "missing_keywords": [],
        }
    }
    at.run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["10/100", "–/10", "–/10"]
    assert not at.subheader
    assert any(t.value == "(no text layer found)" for t in at.text)
    assert any("_None_" in m.value for m in at.markdown)
    assert at.session_state["result"]["report"]["summary"] == "Thin"
