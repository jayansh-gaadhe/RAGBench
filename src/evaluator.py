"""LLM-as-judge evaluation. Each metric is a separate, structured judge call."""
import os
import json
import re
from pydantic import BaseModel, Field, field_validator
from groq import Groq

from .config import RagConfig


class JudgeScore(BaseModel):
    """Validated output schema for a single judge call."""
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

    @field_validator("score", mode="before")
    @classmethod
    def clamp(cls, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, v))


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response, tolerating extra prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"score": 0.0, "reasoning": "Could not parse judge output."}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"score": 0.0, "reasoning": "Malformed judge JSON."}


class Evaluator:
    def __init__(self, config: RagConfig):
        self.config = config
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def _judge(self, rubric: str) -> JudgeScore:
        system = (
            "You are a strict evaluation judge for a RAG system. Score the criterion "
            "from 0.0 to 1.0. Respond with ONLY a JSON object of the form "
            '{"score": <float>, "reasoning": "<one sentence>"}. No other text.'
        )
        resp = self.client.chat.completions.create(
            model=self.config.judge_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": rubric},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        return JudgeScore(**_extract_json(raw))

    # ---- The four metrics ----

    def faithfulness(self, answer: str, contexts: list[str]) -> JudgeScore:
        """Is the answer supported by the retrieved context? (anti-hallucination)"""
        ctx = "\n".join(contexts)
        rubric = (
            "Criterion: FAITHFULNESS. Is every claim in the ANSWER directly supported "
            "by the CONTEXT? Score 1.0 if fully grounded, 0.0 if it contains claims not "
            "in the context (hallucination).\n\n"
            f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}"
        )
        return self._judge(rubric)

    def answer_relevance(self, question: str, answer: str) -> JudgeScore:
        """Does the answer address the question that was asked?"""
        rubric = (
            "Criterion: ANSWER RELEVANCE. Does the ANSWER directly address the QUESTION? "
            "Score 1.0 if fully on-topic, lower if it drifts or is incomplete.\n\n"
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
        )
        return self._judge(rubric)

    def context_precision(self, question: str, contexts: list[str]) -> JudgeScore:
        """Were the retrieved chunks actually relevant to the question? (retrieval quality)"""
        ctx = "\n\n".join(f"[Chunk {i+1}] {c}" for i, c in enumerate(contexts))
        rubric = (
            "Criterion: CONTEXT PRECISION. What fraction of the retrieved CHUNKS are "
            "relevant to answering the QUESTION? Score 1.0 if all chunks are useful, "
            "lower as more irrelevant chunks appear.\n\n"
            f"QUESTION:\n{question}\n\nCHUNKS:\n{ctx}"
        )
        return self._judge(rubric)

    def correctness(self, answer: str, ground_truth: str) -> JudgeScore:
        """Does the answer match the known ground-truth answer?"""
        rubric = (
            "Criterion: CORRECTNESS. Does the ANSWER convey the same factual information "
            "as the GROUND TRUTH? Score 1.0 if factually equivalent, 0.0 if it "
            "contradicts or misses the key fact. Wording differences are fine.\n\n"
            f"GROUND TRUTH:\n{ground_truth}\n\nANSWER:\n{answer}"
        )
        return self._judge(rubric)

    def evaluate_one(self, question: str, answer: str, contexts: list[str],
                     ground_truth: str) -> dict:
        """Run all four metrics for a single Q&A item."""
        return {
            "faithfulness": self.faithfulness(answer, contexts),
            "answer_relevance": self.answer_relevance(question, answer),
            "context_precision": self.context_precision(question, contexts),
            "correctness": self.correctness(answer, ground_truth),
        }
