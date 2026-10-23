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

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally

## Run

```powershell
ollama pull qwen2.5vl
uv sync
uv run streamlit run app.py
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| Model | sidebar / `OLLAMA_MODEL` | `qwen2.5vl` |
| Ollama URL | sidebar / `OLLAMA_HOST` | Ollama default |
| Temperature, pages, feedback language | sidebar | — |

## LangGraph Studio

The graph is also exposed as `cv_agent`:

```powershell
uv run langgraph dev
```

On Windows, if you see log encoding errors, set `$env:PYTHONUTF8 = "1"` first.

Input: `{"pdf_base64": "...", "job_description": "..."}`

## Project structure

```
app.py               Streamlit UI
cv_agent/graph.py    LangGraph definition
cv_agent/pdf_utils.py PDF text/image extraction
cv_agent/schemas.py  Pydantic output schemas
cv_agent/settings.py Frozen `Settings` dataclass (defaults, env vars, limits)
langgraph.json       LangGraph Studio config
```

## Testing

```powershell
uv run pytest            # tests + branch coverage, fails below 100%
uv run tox -q -r            # Python 3.10 – 3.14 matrix (tox-uv)
```

## Privacy

Everything runs locally: your CV is never sent to external services.

## Contributing

Guidelines for GitHub Copilot and contributors are in
[.github/copilot-instructions.md](.github/copilot-instructions.md).

## License

[MIT](LICENSE)
