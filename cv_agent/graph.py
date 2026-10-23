"""LangGraph workflow that analyses a PDF CV and proposes improvement hints.

    START -> extract -> (analyze_layout || analyze_content) -> generate_hints -> END

`analyze_layout` sends rendered page images to the vision model (qwen2.5vl).
`analyze_content` uses the extracted text, or the page images when the PDF
has no text layer (e.g. a scanned CV).
"""

import base64
from dataclasses import dataclass, field
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel

from cv_agent.pdf_utils import extract_text, render_pages
from cv_agent.schemas import ContentReview, CVReport, LayoutReview
from cv_agent.settings import settings

ModelT = TypeVar("ModelT", bound=BaseModel)

MIN_TEXT_CHARS = settings.min_text_chars
MAX_TEXT_CHARS = settings.max_text_chars


@dataclass
class Context:
    model: str = settings.model
    base_url: str = settings.base_url
    temperature: float = settings.temperature
    num_ctx: int = settings.num_ctx
    num_predict: int = settings.num_predict
    max_pages: int = settings.max_pages


@dataclass
class CVState:
    pdf_base64: str
    job_description: str = ""
    language: str = ""
    cv_text: str = ""
    page_images: list[str] = field(default_factory=list)
    layout_review: dict = field(default_factory=dict)
    content_review: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)


def _ctx(runtime: Runtime[Context]) -> Context:
    return runtime.context or Context()


def _llm(ctx: Context) -> ChatOllama:
    return ChatOllama(
        model=ctx.model,
        base_url=ctx.base_url,
        temperature=ctx.temperature,
        num_ctx=ctx.num_ctx,
        num_predict=ctx.num_predict,
        repeat_penalty=settings.repeat_penalty,
    )


def _invoke_structured(
    ctx: Context,
    schema: type[ModelT],
    messages: list,
    retries: int = settings.llm_retries,
) -> ModelT:
    llm = _llm(ctx).with_structured_output(schema, method="json_schema")
    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            result = llm.invoke(messages)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - malformed JSON from the model
            last_error = exc
    raise RuntimeError(
        f"Model failed to return a valid {schema.__name__}"
    ) from last_error


def _image_message(text: str, images: list[str]) -> HumanMessage:
    content: list[str | dict] = [{"type": "text", "text": text}]
    content += [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
        for img in images
    ]
    return HumanMessage(content=content)


def _language_rule(state: CVState) -> str:
    lang = state.language or "the same language as the CV"
    return f"Write every piece of feedback in {lang}."


def _clamp(value: int, high: int) -> int:
    return max(0, min(high, value))


# --- Nodes -----------------------------------------------------------------


def extract(state: CVState, runtime: Runtime[Context]) -> dict:
    pdf_bytes = base64.b64decode(state.pdf_base64)
    return {
        "cv_text": extract_text(pdf_bytes),
        "page_images": render_pages(pdf_bytes, max_pages=_ctx(runtime).max_pages),
    }


def analyze_layout(state: CVState, runtime: Runtime[Context]) -> dict:
    system = SystemMessage(
        content=(
            "You are an expert CV designer and recruiter. You review the VISUAL "
            "presentation of CVs: layout, visual hierarchy, whitespace, fonts, "
            "alignment, consistency, length, readability and ATS-friendliness "
            "(e.g. tables, columns, graphics or icons that parsers can't read). "
            + _language_rule(state)
        )
    )
    human = _image_message(
        "These are the pages of a CV. Assess its visual layout and formatting. "
        "Be specific and reference what you actually see. List each distinct "
        "point once, at most 5 strengths and 6 issues.",
        state.page_images,
    )
    review = _invoke_structured(_ctx(runtime), LayoutReview, [system, human])
    review.score = _clamp(review.score, 10)
    return {"layout_review": review.model_dump()}


def analyze_content(state: CVState, runtime: Runtime[Context]) -> dict:
    system = SystemMessage(
        content=(
            "You are a senior technical recruiter reviewing the CONTENT of a CV: "
            "structure and sections, professional summary, clarity of experience, "
            "quantified achievements, action verbs, skills relevance, education, "
            "gaps, grammar and conciseness. " + _language_rule(state)
        )
    )
    text = state.cv_text
    if len(text) >= MIN_TEXT_CHARS:
        human = HumanMessage(
            content=f"Analyse the content of this CV:\n\n{text[:MAX_TEXT_CHARS]}"
        )
    else:
        # No usable text layer: let the vision model read the pages directly.
        human = _image_message(
            "The CV has no extractable text. Read the pages and analyse its content.",
            state.page_images,
        )
    review = _invoke_structured(_ctx(runtime), ContentReview, [system, human])
    review.score = _clamp(review.score, 10)
    return {"content_review": review.model_dump()}


def generate_hints(state: CVState, runtime: Runtime[Context]) -> dict:
    job = state.job_description.strip()
    job_block = (
        f"\n\nTarget job description:\n{job}\n\nTailor the hints to this role and "
        "list important keywords from it that are missing in the CV."
        if job
        else "\n\nNo job description was provided; leave missing_keywords empty."
    )
    system = SystemMessage(
        content=(
            "You are a career coach. Combine the layout and content reviews into a "
            "final report with 6-12 concrete, actionable hints sorted from highest "
            "to lowest priority. Score the CV from 0 to 100. Avoid generic advice: each hint must refer to this "
            "specific CV. When useful, give a rewritten example line in `example`. "
            + _language_rule(state)
        )
    )
    human = HumanMessage(
        content=(
            f"Layout review:\n{state.layout_review}\n\n"
            f"Content review:\n{state.content_review}\n\n"
            f"CV text:\n{state.cv_text[:MAX_TEXT_CHARS]}"
            f"{job_block}"
        )
    )
    report = _invoke_structured(_ctx(runtime), CVReport, [system, human])
    if 0 < report.overall_score <= 10:  # model sometimes answers on a 0-10 scale
        report.overall_score *= 10
    report.overall_score = _clamp(report.overall_score, 100)
    order = {"High": 0, "Medium": 1, "Low": 2}
    report.hints.sort(key=lambda h: order[h.priority])
    if not job:
        report.missing_keywords = []
    return {"report": report.model_dump()}


def build_graph():
    builder = StateGraph(CVState, context_schema=Context)
    builder.add_node("extract", extract)
    builder.add_node("analyze_layout", analyze_layout)
    builder.add_node("analyze_content", analyze_content)
    builder.add_node("generate_hints", generate_hints)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "analyze_layout")
    builder.add_edge("extract", "analyze_content")
    builder.add_edge(["analyze_layout", "analyze_content"], "generate_hints")
    builder.add_edge("generate_hints", END)
    return builder.compile()


graph = build_graph()
