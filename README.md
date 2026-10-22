# CV Analyzer — LangGraph + Ollama + Streamlit

Upload a PDF CV and get a score, a layout review, a content review and prioritised,
actionable hints to improve it (optionally tailored to a job description).
Runs fully locally with the Ollama vision model `qwen2.5vl`.

## Graph

```
START -> extract -> analyze_layout  (vision: page images)  -> generate_hints -> END
                 \-> analyze_content (text, or images if scanned) -/
```

- `extract` — PyMuPDF extracts the text and renders the first pages to PNG.
- `analyze_layout` — `qwen2.5vl` reviews visual hierarchy, whitespace, fonts, ATS-friendliness.
- `analyze_content` — reviews sections, achievements, wording, skills.
- `generate_hints` — merges both reviews into a 0–100 score, summary, hints and missing keywords.

Every LLM step returns structured output (Pydantic schemas in `cv_agent/schemas.py`).

## Run

```powershell
ollama pull qwen2.5vl
uv sync
uv run streamlit run app.py
```

Model, Ollama URL, temperature, pages and feedback language are configurable in the
sidebar (defaults via `OLLAMA_MODEL` / `OLLAMA_HOST` env vars).

The graph is also exposed to LangGraph Studio as `cv_agent` (`uv run langgraph dev`),
with input `{"pdf_base64": "...", "job_description": "..."}`.
