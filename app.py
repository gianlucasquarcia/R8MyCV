"""Streamlit front-end for the LangGraph CV analyser.

Run with:  uv run streamlit run app.py
"""

import base64

import streamlit as st

from cv_agent.graph import Context, graph

STEP_LABELS = {
    "extract": "📄 Extracted text and rendered pages",
    "analyze_layout": "🎨 Reviewed layout and formatting",
    "analyze_content": "📝 Reviewed content",
    "generate_hints": "💡 Generated improvement hints",
}
ANIMATION_HTML = """
<style>
@keyframes r8-hue {0%{background-position:0% 50%}100%{background-position:300% 50%}}
@keyframes r8-float {0%{transform:translateY(0) rotate(0)}100%{transform:translateY(-120vh) rotate(360deg)}}
@keyframes r8-scan {0%{top:-4px}100%{top:100%}}
@keyframes r8-pulse {0%,100%{opacity:.55}50%{opacity:1}}
.r8-overlay{position:fixed;inset:0;z-index:1000000;pointer-events:none;overflow:hidden;
background:linear-gradient(120deg,rgba(127,90,240,.28),rgba(44,182,125,.28),rgba(255,137,6,.28),
rgba(229,49,112,.28),rgba(127,90,240,.28));background-size:300% 100%;
animation:r8-hue 8s linear infinite}
.r8-scan{position:absolute;left:0;right:0;height:4px;background:#fff;
box-shadow:0 0 12px 2px rgba(255,255,255,.8);animation:r8-scan 2.5s linear infinite}
.r8-item{position:absolute;top:100vh;font-size:2rem;animation:r8-float linear infinite}
.r8-item:nth-of-type(1){left:6%;animation-duration:7s}
.r8-item:nth-of-type(2){left:20%;animation-duration:10s;font-size:3rem}
.r8-item:nth-of-type(3){left:36%;animation-duration:6s;animation-delay:-2s}
.r8-item:nth-of-type(4){left:52%;animation-duration:12s;font-size:3.5rem}
.r8-item:nth-of-type(5){left:68%;animation-duration:8s;animation-delay:-4s}
.r8-item:nth-of-type(6){left:84%;animation-duration:9s;font-size:2.5rem}
.r8-label{position:absolute;top:1.5rem;left:50%;transform:translateX(-50%);padding:.6rem 1.4rem;
border-radius:999px;background:rgba(0,0,0,.6);color:#fff;font-weight:600;
animation:r8-pulse 1.4s ease-in-out infinite}
</style>
<div class="r8-overlay"><div class="r8-scan"></div><span class="r8-item">📄</span>
<span class="r8-item">✨</span><span class="r8-item">🧠</span><span class="r8-item">💡</span>
<span class="r8-item">📝</span><span class="r8-item">⭐</span><div class="r8-label">✨ Analyzing your CV…</div></div>
"""
PRIORITY_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

st.set_page_config(page_title="CV Analyzer", page_icon="📄", layout="wide")
st.title("📄 CV Analyzer")
st.caption(
    "LangGraph agent + local Ollama vision model to review your CV and suggest improvements."
)

with st.sidebar:
    st.header("⚙️ Settings")
    defaults = Context()
    model = st.text_input("Ollama model", defaults.model)
    base_url = st.text_input("Ollama URL", defaults.base_url)
    temperature = st.slider("Temperature", 0.0, 1.0, defaults.temperature, 0.05)
    max_pages = st.number_input("Pages sent to vision model", 1, 6, defaults.max_pages)
    language = st.selectbox(
        "Feedback language",
        ["Same as CV", "English", "Italiano", "Español", "Français", "Deutsch"],
    )

uploaded = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
job_description = st.text_area(
    "Target job description (optional)",
    placeholder="Paste a job posting to get tailored hints and missing keywords…",
    height=150,
)


def report_to_markdown(result: dict) -> str:
    report = result["report"]
    lines = [
        f"# CV Review — score {report['overall_score']}/100",
        "",
        report["summary"],
        "",
        "## Hints",
    ]
    for i, h in enumerate(report["hints"], 1):
        lines.append(f"{i}. **[{h['priority']}] {h['category']}** — {h['issue']}")
        lines.append(f"   - Suggestion: {h['suggestion']}")
        if h.get("example"):
            lines.append(f"   - Example: _{h['example']}_")
    if report.get("missing_keywords"):
        lines += ["", "## Missing keywords", ", ".join(report["missing_keywords"])]
    return "\n".join(lines)


def render_list(title: str, items: list[str]) -> None:
    st.markdown(f"**{title}**")
    if items:
        st.markdown("\n".join(f"- {i}" for i in items))
    else:
        st.markdown("_None_")


if uploaded and st.button("🔍 Analyze CV", type="primary"):
    context = Context(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_pages=int(max_pages),
    )
    inputs = {
        "pdf_base64": base64.b64encode(uploaded.getvalue()).decode("ascii"),
        "job_description": job_description,
        "language": "" if language == "Same as CV" else language,
    }
    result: dict = {}
    loader = st.empty()
    loader.markdown(ANIMATION_HTML, unsafe_allow_html=True)
    with st.status("Analyzing your CV…", expanded=True) as status:
        try:
            for update in graph.stream(inputs, context=context, stream_mode="updates"):
                for node, output in update.items():
                    result.update(output or {})
                    st.write(STEP_LABELS.get(node, node))
            status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
            status.update(label="Analysis failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            st.info(
                f"Make sure Ollama is running at {base_url} and `{model}` is pulled."
            )
    loader.empty()
    if "report" in result:
        st.session_state["result"] = result
        st.session_state["file_name"] = uploaded.name

saved: dict | None = st.session_state.get("result")
if saved:
    report = saved["report"]
    layout = saved.get("layout_review", {})
    content = saved.get("content_review", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall score", f"{report['overall_score']}/100")
    c2.metric("Content", f"{content.get('score', '–')}/10")
    c3.metric("Layout", f"{layout.get('score', '–')}/10")
    st.progress(report["overall_score"] / 100)
    st.info(report["summary"])

    tab_hints, tab_content, tab_layout, tab_cv = st.tabs(
        ["💡 Hints", "📝 Content review", "🎨 Layout review", "📄 CV"]
    )

    with tab_hints:
        for h in report["hints"]:
            with st.expander(
                f"{PRIORITY_ICON[h['priority']]} {h['category']} — {h['issue']}"
            ):
                st.markdown(f"**Suggestion:** {h['suggestion']}")
                if h.get("example"):
                    st.markdown("**Example:**")
                    st.code(h["example"], language=None, wrap_lines=True)
        if report.get("missing_keywords"):
            st.subheader("🔑 Missing keywords")
            st.write(" · ".join(f"`{k}`" for k in report["missing_keywords"]))
        st.download_button(
            "⬇️ Download report (Markdown)",
            report_to_markdown(saved),
            file_name=f"{st.session_state.get('file_name', 'cv')}_review.md",
            mime="text/markdown",
        )

    with tab_content:
        if content.get("candidate_name") or content.get("current_role"):
            st.markdown(
                f"**{content.get('candidate_name', '')}** — {content.get('current_role', '')}"
            )
        col_a, col_b = st.columns(2)
        with col_a:
            render_list("✅ Strengths", content.get("strengths", []))
            render_list("📑 Detected sections", content.get("detected_sections", []))
        with col_b:
            render_list("⚠️ Weaknesses", content.get("weaknesses", []))
            render_list("❓ Missing sections", content.get("missing_sections", []))

    with tab_layout:
        col_a, col_b = st.columns(2)
        with col_a:
            render_list("✅ Strengths", layout.get("strengths", []))
        with col_b:
            render_list("⚠️ Issues", layout.get("issues", []))

    with tab_cv:
        images = saved.get("page_images", [])
        if images:
            cols = st.columns(len(images))
            for i, (col, img) in enumerate(zip(cols, images), 1):
                col.image(base64.b64decode(img), caption=f"Page {i}", width="stretch")
        with st.expander("Extracted text"):
            st.text(saved.get("cv_text") or "(no text layer found)")
