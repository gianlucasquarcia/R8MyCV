import pytest
from pydantic import ValidationError

from cv_agent.schemas import ContentReview, CVReport, Hint, LayoutReview


def hint(issue: str, priority: str = "High") -> dict:
    return {
        "category": "Content",
        "priority": priority,
        "issue": issue,
        "suggestion": "Fix it",
    }


def test_lists_are_deduplicated_and_truncated() -> None:
    review = LayoutReview(
        score=7,
        strengths=["Clean", "clean", " Clean ", "", "A", "B", "C", "D", "E", "F"],
        issues=[],
    )
    assert review.strengths == ["Clean", "A", "B", "C", "D"]
    assert review.issues == []


def test_non_list_values_are_left_to_pydantic() -> None:
    with pytest.raises(ValidationError):
        LayoutReview(score=1, strengths="not a list")


def test_content_review_defaults() -> None:
    review = ContentReview(score=5)
    assert review.candidate_name == ""
    assert review.detected_sections == []


def test_report_dedupes_hints_and_caps_at_twelve() -> None:
    hints = [hint("Same"), hint(" SAME ")] + [hint(f"Issue {i}") for i in range(20)]
    report = CVReport(overall_score=80, summary="ok", hints=hints)
    assert len(report.hints) == 12
    assert report.hints[0].issue == "Same"
    assert report.missing_keywords == []


def test_report_accepts_hint_instances() -> None:
    instance = Hint(**hint("Instance"))
    report = CVReport(overall_score=1, summary="s", hints=[instance])
    assert report.hints == [instance]


def test_report_rejects_non_list_hints() -> None:
    with pytest.raises(ValidationError):
        CVReport(overall_score=1, summary="s", hints="nope")
