"""The RAG pipeline being evaluated: ChromaDB for retrieval, Groq for generation."""
import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

from .config import RagConfig


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple character-based chunker with overlap.

    Hand-rolled on purpose so the chunking behaviour is transparent and tunable —
    the harness's whole point is measuring how chunking affects quality.
    """
    text = text.strip()
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


class RagPipeline:
    def __init__(self, config: RagConfig):
        self.config = config
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

        # In-memory Chroma client; fresh collection per pipeline so configs
        # don't contaminate each other.
        self._chroma = chromadb.EphemeralClient()
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )
        # Unique name so two configs can coexist in one process.
        self.collection = self._chroma.create_collection(
            name=f"corpus_{abs(hash(config.label()))}",
            embedding_function=self._embed_fn,
        )

    def index(self) -> int:
        """Read the corpus, chunk it, embed and store in ChromaDB."""
        with open(self.config.corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
        self.collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
        )
        return len(chunks)

    def retrieve(self, question: str) -> list[str]:
        """Return the top_k most relevant chunks for a question."""
        results = self.collection.query(
            query_texts=[question],
            n_results=self.config.top_k,
        )
        return results["documents"][0]

    def generate(self, question: str, context_chunks: list[str]) -> str:
        """Answer the question grounded ONLY in the retrieved context."""
        context = "\n\n".join(f"[Chunk {i+1}] {c}" for i, c in enumerate(context_chunks))
        prompt = (
            "You are a helpful assistant. Answer the question using ONLY the context "
            "below. If the answer is not in the context, say you don't know. Do not "
            "use outside knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        resp = self.client.chat.completions.create(
            model=self.config.gen_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

    def answer(self, question: str) -> dict:
        """Full RAG step: retrieve then generate. Returns answer + context for eval."""
        chunks = self.retrieve(question)
        answer = self.generate(question, chunks)
        return {"answer": answer, "contexts": chunks}
