"""Local BGE embeddings for financial-document retrieval."""
from __future__ import annotations
from typing import Sequence
from sentence_transformers import SentenceTransformer


class BGEEmbeddings:
    """Sentence-Transformers adapter compatible with Chroma embedding APIs."""

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Convert document chunks into vectors."""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Convert a user search query into a BGE query vector."""
        query = self.QUERY_INSTRUCTION + text.strip()
        vector = self.model.encode(
            query,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()
