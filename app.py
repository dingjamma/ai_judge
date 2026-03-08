"""
Phase 5: Streamlit Dashboard
Three tabs: Browse Cases, Accuracy Stats, Try It (live verdict).
"""

import json
import os
from pathlib import Path

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Inject Streamlit Cloud secrets into environment (no-op locally)
try:
    import streamlit as _st
    for _k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
        if _k in _st.secrets and not os.environ.get(_k):
            os.environ[_k] = _st.secrets[_k]
except Exception:
    pass

# Add src/ to path so we can import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from db import get_all_cases, get_judged_cases, init_db

st.set_page_config(page_title="AI Judge — SCOTUS Backtester", layout="wide")

init_db()


# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_cases():
    return get_judged_cases()


def verdict_badge(verdict: str | None) -> str:
    colors = {"affirmed": "green", "reversed": "red", "vacated": "orange"}
    c = colors.get(verdict or "", "gray")
    return f":{c}[{verdict or 'unknown'}]"


# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("AI Judge")
st.sidebar.caption("SCOTUS Verdict Backtester")

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Browse Cases", "Accuracy Stats", "Try It"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Browse Cases
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Browse Cases")
    cases = load_cases()

    if not cases:
        st.info("No judged cases yet. Run `python src/judge.py --all` first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            search = st.text_input("Search by case name", "")
        with col2:
            case_types = ["All"] + sorted({c["case_type"] or "unknown" for c in cases})
            type_filter = st.selectbox("Case type", case_types)
        with col3:
            match_filter = st.selectbox("Verdict match", ["All", "Correct", "Incorrect"])

        filtered = cases
        if search:
            filtered = [c for c in filtered if search.lower() in (c["name"] or "").lower()]
        if type_filter != "All":
            filtered = [c for c in filtered if c["case_type"] == type_filter]
        if match_filter == "Correct":
            filtered = [c for c in filtered if c["match"]]
        elif match_filter == "Incorrect":
            filtered = [c for c in filtered if c["match"] is False]

        st.metric("Cases shown", len(filtered))

        for c in filtered[:50]:
            with st.expander(f"{c['name']} ({c.get('year', '?')})"):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Actual", c["actual_verdict"] or "unknown")
                col_b.metric("AI Verdict", c["ai_verdict"] or "pending")
                col_c.metric("Confidence", f"{c['ai_confidence']:.0%}" if c["ai_confidence"] else "—")

                st.markdown(f"**Reasoning:** {c.get('ai_reasoning') or '—'}")
                st.markdown(f"**Fairness score:** {c.get('fairness_score') or '—'}")
                if c.get("fairness_notes"):
                    st.caption(c["fairness_notes"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Accuracy Stats
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Accuracy Statistics")
    cases = load_cases()

    if not cases:
        st.info("No judged cases yet.")
    else:
        total = len(cases)
        correct = sum(1 for c in cases if c["match"])
        accuracy = correct / total if total else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Accuracy", f"{accuracy:.1%}")
        m2.metric("Cases Judged", total)
        m3.metric("Correct", correct)

        # Accuracy by decade
        from collections import defaultdict
        by_decade: dict = defaultdict(lambda: {"correct": 0, "total": 0})
        for c in cases:
            if c.get("year"):
                dec = f"{(c['year'] // 10) * 10}s"
                by_decade[dec]["total"] += 1
                if c["match"]:
                    by_decade[dec]["correct"] += 1

        decade_df = [
            {"Decade": d, "Accuracy": v["correct"] / v["total"]}
            for d, v in sorted(by_decade.items())
            if v["total"] > 0
        ]
        if decade_df:
            fig = px.line(decade_df, x="Decade", y="Accuracy", title="Accuracy by Decade",
                          markers=True, range_y=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        # Accuracy by case type
        by_type: dict = defaultdict(lambda: {"correct": 0, "total": 0})
        for c in cases:
            t = c.get("case_type") or "unknown"
            by_type[t]["total"] += 1
            if c["match"]:
                by_type[t]["correct"] += 1

        type_df = [
            {"Case Type": t, "Accuracy": v["correct"] / v["total"], "Count": v["total"]}
            for t, v in sorted(by_type.items())
            if v["total"] > 0
        ]
        if type_df:
            fig2 = px.bar(type_df, x="Case Type", y="Accuracy", color="Count",
                          title="Accuracy by Case Type", range_y=[0, 1])
            st.plotly_chart(fig2, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Try It (live verdict)
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Try It — Live Verdict")
    st.caption("Paste hypothetical case facts below and get an AI verdict.")

    facts_input = st.text_area("Case Facts", height=200,
                               placeholder="Describe the facts of the case...")

    if st.button("Render Verdict", type="primary") and facts_input.strip():
        try:
            import boto3
            from embed import embed_text, get_bedrock_client, FAISS_INDEX_PATH, CASE_MAP_PATH
            from judge import load_index_and_map, retrieve_similar, format_precedents, call_claude, PROMPT_TEMPLATE
            import numpy as np, faiss

            with st.spinner("Thinking..."):
                client = get_bedrock_client()
                index, case_map = load_index_and_map()
                similar = retrieve_similar(client, index, case_map, facts_input, "__live__")
                precedents_str = format_precedents(similar)
                prompt = PROMPT_TEMPLATE.format(
                    facts=facts_input[:3000],
                    retrieved_cases=precedents_str,
                )
                result = call_claude(client, prompt)

            verdict_color = {"affirmed": "green", "reversed": "red", "vacated": "orange"}
            v = result.get("verdict", "unknown")
            st.success(f"Verdict: **{v.upper()}**")

            c1, c2 = st.columns(2)
            c1.metric("Confidence", f"{result.get('confidence', 0):.0%}")
            c2.metric("Fairness Score", f"{result.get('fairness_score', 0):.0%}")

            st.markdown(f"**Reasoning:** {result.get('reasoning', '—')}")
            if result.get("fairness_notes"):
                st.info(f"Fairness notes: {result['fairness_notes']}")

            if similar:
                st.subheader("Top Precedents Used")
                for c in similar:
                    st.markdown(f"- **{c.get('name')}** ({c.get('year')}) — {c.get('actual_verdict')}")

        except FileNotFoundError:
            st.error("FAISS index not found. Run `python src/embed.py` first.")
        except Exception as e:
            st.error(f"Error: {e}")
