# Copilot instructions for R8MyCV

CV analyzer built with LangGraph + Ollama (`qwen2.5vl`) + Streamlit. Python >= 3.10, managed with `uv`.

## Layout
- `app.py` — Streamlit UI (sidebar config, PDF upload, results rendering).
- `cv_agent/graph.py` — LangGraph graph: `extract -> analyze_layout | analyze_content -> generate_hints`.
- `cv_agent/pdf_utils.py` — PyMuPDF text extraction and page-to-PNG rendering.
- `cv_agent/schemas.py` — Pydantic schemas for all structured LLM output.
- `langgraph.json` — exposes the graph as `cv_agent` for LangGraph Studio.

- `tests/` — pytest suite (LLM and Ollama are faked; Streamlit UI via `AppTest`).
- `tox.ini` — Python 3.10–3.14 matrix run with tox-uv.

## Commands
- Test: `uv run pytest` (100% branch coverage is enforced) and `uv run tox -p` for the matrix.
- Install: `uv sync`
- Run UI: `uv run streamlit run app.py`
- Studio: `uv run langgraph dev`
- Add deps with `uv add <pkg>` (keeps `pyproject.toml` and `uv.lock` in sync).

## Conventions
- Every LLM step must return structured output via a Pydantic schema in `cv_agent/schemas.py`; do not parse free text.
- Keep the graph runnable fully locally; do not add cloud LLM calls or send CV data to third parties.
- Config comes from the sidebar with defaults from `OLLAMA_MODEL` / `OLLAMA_HOST` env vars.
- Keep graph state input compatible with `{"pdf_base64": "...", "job_description": "..."}`.
- Use type hints, small focused functions, and no unnecessary comments.
- Never commit CVs, personal data or secrets.
- Update `README.md` when changing the graph, config or run steps.
