"""Optional Streamlit dashboard for the eval harness.

Run with:  streamlit run src/dashboard.py
"""
import os
import sys

# Allow running as a script from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from src.config import RagConfig
from src.main import run_config, METRICS

st.set_page_config(page_title="RAG Evaluation Harness", layout="wide")
st.title("RAG Evaluation Harness")
st.caption("ChromaDB + Groq · LLM-as-judge metrics")

with st.sidebar:
    st.header("Config")
    chunk_size = st.slider("Chunk size", 100, 1000, 500, step=50)
    overlap = st.slider("Chunk overlap", 0, 200, 50, step=10)
    top_k = st.slider("Top-k retrieved", 1, 6, 3)
    run = st.button("Run evaluation", type="primary")
    if "GROQ_API_KEY" not in os.environ:
        st.warning("Set GROQ_API_KEY in your environment before running.")

if run:
    config = RagConfig(chunk_size=chunk_size, chunk_overlap=overlap, top_k=top_k)
    with st.spinner("Indexing, answering, and judging..."):
        df = run_config(config, verbose=False)

    st.subheader("Average scores")
    means = df[METRICS].mean()
    cols = st.columns(len(METRICS) + 1)
    for col, m in zip(cols, METRICS):
        col.metric(m.replace("_", " ").title(), f"{means[m]:.0%}")
    cols[-1].metric("Overall", f"{means.mean():.0%}")

    st.bar_chart(means)

    st.subheader("Per-question results")
    show = df.copy()
    show["avg"] = show[METRICS].mean(axis=1)
    st.dataframe(
        show.sort_values("avg").style.background_gradient(
            subset=METRICS + ["avg"], cmap="RdYlGn", vmin=0, vmax=1
        ),
        use_container_width=True,
    )
else:
    st.info("Set your config in the sidebar and click **Run evaluation**.")
