"""Central configuration. Change these to test how RAG quality responds."""
from dataclasses import dataclass


@dataclass
class RagConfig:
    # --- Retrieval / chunking knobs (the things you tune and measure) ---
    chunk_size: int = 500          # characters per chunk
    chunk_overlap: int = 50        # overlap between consecutive chunks
    top_k: int = 3                 # number of chunks retrieved per question

    # --- Models (Groq) ---
    # Generation model: answers the questions.
    gen_model: str = "llama-3.3-70b-versatile"
    # Judge model: scores the answers. Kept separate on purpose.
    judge_model: str = "llama-3.3-70b-versatile"

    # --- Embeddings ---
    # ChromaDB's default embedding function (all-MiniLM-L6-v2) is used so the
    # project runs with zero extra API cost for embeddings.
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Paths ---
    corpus_path: str = "data/corpus.txt"
    testset_path: str = "data/testset.json"

    def label(self) -> str:
        return f"chunk={self.chunk_size}/overlap={self.chunk_overlap}/k={self.top_k}"


# Two configs used by the --compare mode to demonstrate measurable differences.
CONFIG_A = RagConfig(chunk_size=500, chunk_overlap=50, top_k=3)
CONFIG_B = RagConfig(chunk_size=200, chunk_overlap=20, top_k=2)
