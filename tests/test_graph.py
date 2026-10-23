import base64
import importlib
from types import SimpleNamespace
from typing import Optional

import pytest
from langchain_core.messages import HumanMessage

from cv_agent.graph import (
    Context,
    CVState,
    _clamp,
    _image_message,
    _invoke_structured,
    _language_rule,
    _llm,
    analyze_content,
    analyze_layout,
    extract,
    generate_hints,
)
from cv_agent.schemas import ContentReview, CVReport, LayoutReview

graph_module = importlib.import_module("cv_agent.graph")

class FakeLLM:
    """Stands in for ChatOllama; responses are queued per schema."""

    def __init__(self, queues: dict) -> None:
        self.queues = queues
        self.calls: list = []
        self.methods: list = []

    def with_structured_output(self, schema: type, method: str):
        self.methods.append(method)
        return SimpleNamespace(
            schema=schema, invoke=lambda messages: self._invoke(schema, messages)
        )

    def _invoke(self, schema: type, messages: list):
        self.calls.append((schema, messages))
        response = self.queues[schema].pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def messages_for(self, schema: type) -> list:
        return next(m for s, m in self.calls if s is schema)


@pytest.fixture
def use_llm(monkeypatch: pytest.MonkeyPatch):
    def install(queues: dict) -> FakeLLM:
        fake = FakeLLM({schema: list(items) for schema, items in queues.items()})
        monkeypatch.setattr(graph_module, "_llm", lambda ctx: fake)
        return fake

    return install


def runtime(context: Optional[Context] = None) -> SimpleNamespace:
    return SimpleNamespace(context=context)


def report_payload(**overrides) -> dict:
    payload = {
        "overall_score": 72,
        "summary": "Solid CV",
        "hints": [
            {
                "category": "Layout",
                "priority": "Low",
                "issue": "Margins",
                "suggestion": "Widen",
            },
            {
                "category": "Content",
                "priority": "High",
                "issue": "No metrics",
                "suggestion": "Quantify",
            },
            {
                "category": "Impact",
                "priority": "Medium",
                "issue": "Weak verbs",
                "suggestion": "Use action verbs",
            },
        ],
        "missing_keywords": ["Kubernetes"],
    }
    payload.update(overrides)
    return payload


def test_clamp() -> None:
    assert (_clamp(-3, 10), _clamp(5, 10), _clamp(50, 10)) == (0, 5, 10)


def test_language_rule() -> None:
    assert "the same language as the CV" in _language_rule(CVState(pdf_base64=""))
    assert "Italiano" in _language_rule(CVState(pdf_base64="", language="Italiano"))


def test_image_message_embeds_images() -> None:
    message = _image_message("look", ["abc", "def"])
    assert isinstance(message, HumanMessage)
    assert message.content[0] == {"type": "text", "text": "look"}
    assert message.content[1]["image_url"]["url"] == "data:image/png;base64,abc"
    assert len(message.content) == 3


def test_llm_uses_context() -> None:
    llm = _llm(Context(model="m", base_url="http://h:1", temperature=0.5))
    assert llm.model == "m"
    assert llm.base_url == "http://h:1"
    assert llm.temperature == 0.5


def test_invoke_structured_returns_model_instance(use_llm) -> None:
    review = LayoutReview(score=4)
    fake = use_llm({LayoutReview: [review]})
    assert _invoke_structured(Context(), LayoutReview, ["m"]) is review
    assert fake.methods == ["json_schema"]


def test_invoke_structured_validates_dicts(use_llm) -> None:
    use_llm({LayoutReview: [{"score": 6}]})
    assert _invoke_structured(Context(), LayoutReview, []).score == 6


def test_invoke_structured_retries_after_errors(use_llm) -> None:
    fake = use_llm({LayoutReview: [ValueError("bad json"), {"score": 3}]})
    assert _invoke_structured(Context(), LayoutReview, []).score == 3
    assert len(fake.calls) == 2


def test_invoke_structured_raises_after_retries(use_llm) -> None:
    use_llm({LayoutReview: [ValueError("one"), ValueError("two")]})
    with pytest.raises(RuntimeError, match="LayoutReview") as info:
        _invoke_structured(Context(), LayoutReview, [], retries=1)
    assert isinstance(info.value.__cause__, ValueError)


def test_invoke_structured_raises_on_unusable_result(use_llm) -> None:
    use_llm({LayoutReview: [None]})
    with pytest.raises(RuntimeError) as info:
        _invoke_structured(Context(), LayoutReview, [], retries=0)
    assert info.value.__cause__ is None


def test_extract_node(pdf_bytes: bytes) -> None:
    state = CVState(pdf_base64=base64.b64encode(pdf_bytes).decode())
    out = extract(state, runtime(Context(max_pages=2)))
    assert "Jane Doe" in out["cv_text"]
    assert len(out["page_images"]) == 2


def test_extract_node_defaults_context_when_missing(pdf_bytes: bytes) -> None:
    state = CVState(pdf_base64=base64.b64encode(pdf_bytes).decode())
    assert len(extract(state, runtime())["page_images"]) == 3


def test_analyze_layout_clamps_score(use_llm) -> None:
    fake = use_llm({LayoutReview: [{"score": 99, "strengths": ["a"], "issues": ["b"]}]})
    out = analyze_layout(CVState(pdf_base64="", page_images=["img"]), runtime())
    assert out["layout_review"]["score"] == 10
    assert fake.messages_for(LayoutReview)[1].content[1]["type"] == "image_url"


def test_analyze_content_uses_text_when_available(use_llm) -> None:
    fake = use_llm({ContentReview: [{"score": -4}]})
    state = CVState(pdf_base64="", cv_text="x" * 300)
    out = analyze_content(state, runtime())
    assert out["content_review"]["score"] == 0
    assert isinstance(fake.messages_for(ContentReview)[1].content, str)


def test_analyze_content_falls_back_to_images(use_llm) -> None:
    fake = use_llm({ContentReview: [{"score": 5}]})
    state = CVState(pdf_base64="", cv_text="short", page_images=["img"])
    analyze_content(state, runtime())
    assert isinstance(fake.messages_for(ContentReview)[1].content, list)


def test_generate_hints_with_job_description(use_llm) -> None:
    fake = use_llm({CVReport: [report_payload()]})
    state = CVState(pdf_base64="", job_description="  Kubernetes engineer ")
    report = generate_hints(state, runtime())["report"]
    assert [h["priority"] for h in report["hints"]] == ["High", "Medium", "Low"]
    assert report["missing_keywords"] == ["Kubernetes"]
    assert "Target job description" in fake.messages_for(CVReport)[1].content


def test_generate_hints_without_job_description(use_llm) -> None:
    use_llm({CVReport: [report_payload()]})
    report = generate_hints(CVState(pdf_base64=""), runtime())["report"]
    assert report["missing_keywords"] == []


@pytest.mark.parametrize(
    ("raw", "expected"), [(8, 80), (0, 0), (10, 100), (11, 11), (250, 100), (-5, 0)]
)
def test_generate_hints_normalises_score(use_llm, raw: int, expected: int) -> None:
    use_llm({CVReport: [report_payload(overall_score=raw)]})
    report = generate_hints(CVState(pdf_base64=""), runtime())["report"]
    assert report["overall_score"] == expected


def test_full_graph_end_to_end(use_llm, pdf_bytes: bytes) -> None:
    use_llm(
        {
            LayoutReview: [{"score": 7, "strengths": ["Clean"], "issues": []}],
            ContentReview: [{"score": 6, "candidate_name": "Jane Doe"}],
            CVReport: [report_payload()],
        }
    )
    inputs = {
        "pdf_base64": base64.b64encode(pdf_bytes).decode(),
        "job_description": "Kubernetes",
        "language": "English",
    }
    result = graph_module.graph.invoke(inputs, context=Context(max_pages=1))
    assert result["report"]["overall_score"] == 72
    assert result["layout_review"]["score"] == 7
    assert result["content_review"]["candidate_name"] == "Jane Doe"
    assert len(result["page_images"]) == 1
