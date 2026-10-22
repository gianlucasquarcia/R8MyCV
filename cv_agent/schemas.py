"""Structured outputs produced by the CV analysis graph."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field


def _bounded(max_items: int) -> BeforeValidator:
    """Deduplicate and truncate lists: small local models tend to loop."""

    def clean(value: Any) -> Any:
        if not isinstance(value, list):
            return value
        seen: dict[str, None] = {}
        for item in value:
            text = str(item).strip()
            if text and text.lower() not in {s.lower() for s in seen}:
                seen[text] = None
        return list(seen)[:max_items]

    return BeforeValidator(clean)


def _dedupe_hints(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    seen: set[str] = set()
    out: list[Any] = []
    for hint in value:
        raw = hint.get("issue", "") if isinstance(hint, dict) else hint
        key = str(raw).strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(hint)
    return out[:12]


List5 = Annotated[list[str], _bounded(5)]
List6 = Annotated[list[str], _bounded(6)]
List10 = Annotated[list[str], _bounded(10)]
List15 = Annotated[list[str], _bounded(15)]


def _list_field(max_items: int, description: str = "") -> Any:
    return Field(default_factory=list, max_length=max_items, description=description)


class LayoutReview(BaseModel):
    score: int = Field(description="Visual/layout quality score from 0 to 10")
    strengths: List5 = _list_field(5, "Distinct positive aspects of the layout")
    issues: List6 = _list_field(
        6, "Distinct, concrete layout, readability or formatting problems"
    )


class ContentReview(BaseModel):
    candidate_name: str = Field(default="", description="Candidate name if found")
    current_role: str = Field(default="", description="Most recent job title")
    detected_sections: List10 = _list_field(10, "Section headings present in the CV")
    missing_sections: List6 = _list_field(
        6, "Important sections that are absent (e.g. Summary, Projects)"
    )
    strengths: List6 = _list_field(6)
    weaknesses: List6 = _list_field(6)
    score: int = Field(description="Content quality score from 0 to 10")


class Hint(BaseModel):
    category: Literal[
        "Content", "Impact", "Structure", "Layout", "Keywords/ATS", "Language"
    ]
    priority: Literal["High", "Medium", "Low"]
    issue: str = Field(description="What is wrong or could be improved")
    suggestion: str = Field(description="Concrete, actionable improvement")
    example: str = Field(
        default="", description="Optional rewritten example text for the CV"
    )


class CVReport(BaseModel):
    overall_score: int = Field(description="Overall CV score from 0 to 100")
    summary: str = Field(description="Short overall assessment (2-4 sentences)")
    hints: Annotated[list[Hint], BeforeValidator(_dedupe_hints)] = Field(
        max_length=12, description="6-12 distinct, prioritised improvement hints"
    )
    missing_keywords: List15 = _list_field(
        15, "Relevant keywords from the job description missing in the CV"
    )
