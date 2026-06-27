# RAG Evaluation Harness

A from-scratch evaluation harness for Retrieval-Augmented Generation (RAG) systems.
It builds a small RAG pipeline (ChromaDB + Groq) and then **measures its quality**
using an LLM-as-judge approach across four metrics:

- **Faithfulness** — is the answer grounded in the retrieved context, or hallucinated?
- **Answer Relevance** — does the answer actually address the question?
- **Context Precision** — were the retrieved chunks actually useful?
- **Correctness** — does the answer match the ground-truth answer?

The harness lets you change a config (chunk size, top-k, model) and **prove**
whether the change made the RAG system better or worse.


## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Get a free Groq API key from https://console.groq.com and set it:

```bash
export GROQ_API_KEY="your_key_here"
```

3. Run the full evaluation:

```bash
python -m src.main
```

4. (Optional) Compare two configs to see which is better:

```bash
python -m src.main --compare
```

5. (Optional) Launch the dashboard:

```bash
streamlit run src/dashboard.py
```

---

## Project structure

```
rag-eval-harness/
├── README.md
├── requirements.txt
├── data/
│   ├── corpus.txt          # the knowledge base (private documents)
│   └── testset.json        # questions + ground-truth answers
└── src/
    ├── config.py           # tunable settings (chunk size, top_k, models)
    ├── rag_pipeline.py      # ChromaDB + Groq RAG system
    ├── evaluator.py        # LLM-as-judge metrics
    ├── main.py             # runs evaluation, prints report
    └── dashboard.py        # optional Streamlit UI
```

## How the metrics work

Each metric is computed by a **second LLM call** (the "judge") that receives a
strict rubric and returns a structured score (0–1) plus a short justification.
Scores are validated so the report is always machine-readable.

| Metric | Inputs to judge | What a low score means |
|---|---|---|
| Faithfulness | answer + retrieved context | model hallucinated / went beyond the context |
| Answer Relevance | question + answer | answer drifted off-topic |
| Context Precision | question + retrieved chunks | retriever fetched irrelevant chunks |
| Correctness | answer + ground truth | answer is factually wrong |

Separating **context precision** (retrieval quality) from **faithfulness**
(generation quality) lets you diagnose *where* a failure happened — the single
most useful thing this harness does.
