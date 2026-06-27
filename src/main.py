"""Entry point: run the RAG pipeline over the test set and score every answer."""
import os
import sys
import json
import argparse

import pandas as pd

from .config import RagConfig, CONFIG_A, CONFIG_B
from .rag_pipeline import RagPipeline
from .evaluator import Evaluator

from dotenv import load_dotenv

load_dotenv()

METRICS = ["faithfulness", "answer_relevance", "context_precision", "correctness"]


def load_testset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_config(config: RagConfig, verbose: bool = True) -> pd.DataFrame:
    """Index, answer, and evaluate for one config. Returns a per-question DataFrame."""
    if "GROQ_API_KEY" not in os.environ:
        sys.exit("ERROR: set GROQ_API_KEY (export GROQ_API_KEY='...').")

    pipeline = RagPipeline(config)
    n_chunks = pipeline.index()
    evaluator = Evaluator(config)
    testset = load_testset(config.testset_path)

    if verbose:
        print(f"\n=== Config: {config.label()} ===")
        print(f"Indexed {n_chunks} chunks. Evaluating {len(testset)} questions...\n")

    rows = []
    for i, item in enumerate(testset, 1):
        q, gt = item["question"], item["ground_truth"]
        result = pipeline.answer(q)
        scores = evaluator.evaluate_one(q, result["answer"], result["contexts"], gt)

        row = {"question": q, "answer": result["answer"]}
        for m in METRICS:
            row[m] = scores[m].score
        rows.append(row)

        if verbose:
            avg = sum(scores[m].score for m in METRICS) / len(METRICS)
            print(f"[{i}/{len(testset)}] avg={avg:.2f}  {q[:55]}...")

    return pd.DataFrame(rows)


def print_report(df: pd.DataFrame, label: str):
    print(f"\n{'='*60}\nRESULTS — {label}\n{'='*60}")
    means = df[METRICS].mean()
    for m in METRICS:
        bar = "#" * int(means[m] * 20)
        print(f"  {m:<20} {means[m]:.2%}  {bar}")
    print(f"  {'OVERALL':<20} {means.mean():.2%}")

    # Surface the weakest answers — the actionable part.
    df = df.copy()
    df["avg"] = df[METRICS].mean(axis=1)
    worst = df.nsmallest(3, "avg")
    print("\n  Weakest answers (investigate these):")
    for _, r in worst.iterrows():
        print(f"   - [{r['avg']:.2f}] {r['question'][:60]}")


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Harness")
    parser.add_argument("--compare", action="store_true",
                        help="Run two configs and compare them.")
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="Save per-question results to a CSV.")
    args = parser.parse_args()

    if args.compare:
        df_a = run_config(CONFIG_A)
        print_report(df_a, CONFIG_A.label())
        df_b = run_config(CONFIG_B)
        print_report(df_b, CONFIG_B.label())

        print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
        comp = pd.DataFrame({
            CONFIG_A.label(): df_a[METRICS].mean(),
            CONFIG_B.label(): df_b[METRICS].mean(),
        })
        comp["delta"] = comp.iloc[:, 0] - comp.iloc[:, 1]
        print(comp.to_string(float_format=lambda x: f"{x:.2%}"))
        winner = CONFIG_A.label() if comp.iloc[:, 0].mean() >= comp.iloc[:, 1].mean() else CONFIG_B.label()
        print(f"\n  Winner: {winner}")
        if args.save:
            df_a.to_csv(args.save, index=False)
            print(f"  Saved Config A results to {args.save}")
    else:
        df = run_config(CONFIG_A)
        print_report(df, CONFIG_A.label())
        if args.save:
            df.to_csv(args.save, index=False)
            print(f"\nSaved results to {args.save}")


if __name__ == "__main__":
    main()
