
from pathlib import Path
import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import BGEEmbeddings
from config.settings import PROJECT_ROOT


# Persistent database directory
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Collection name
COLLECTION_NAME = "financial_documents"


def generate_chunk_id(document: Document) -> str:
    """
    Generate a deterministic ID for each chunk.

    The same chunk will receive the same ID if ingested again.
    """
    identity = "|".join([
        str(document.metadata.get("source", "")),
        str(document.metadata.get("page", "")),
        str(document.metadata.get("chunk_index", "")),
        document.page_content
    ])

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def sanitize_metadata(metadata: dict) -> dict:
    """
    Chroma metadata supports scalar values:
    strings, integers, floats, and booleans.
    """
    clean_metadata = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            clean_metadata[key] = value
        else:
            clean_metadata[key] = str(value)

    return clean_metadata


class FinancialVectorStore:

    def __init__(
        self,
        embedding_model: BGEEmbeddings,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str | Path = CHROMA_DIR,
    ):
        self.embedding_model = embedding_model

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=str(persist_directory),
            collection_metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Generate embeddings and store documents in ChromaDB.

        IDs are deterministic, so re-ingesting an unchanged chunk
        updates the existing record instead of creating a duplicate.
        """
        if not documents:
            return []

        ids = [generate_chunk_id(doc) for doc in documents]

        cleaned_documents = [
            Document(
                page_content=doc.page_content,
                metadata=sanitize_metadata(doc.metadata)
            )
            for doc in documents
        ]

        self.vector_store.add_documents(
            documents=cleaned_documents,
            ids=ids
        )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None
    ) -> list[Document]:

        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filters
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None
    ):
        """
        Return documents along with Chroma distance scores.
        For cosine distance, lower is more similar.
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filters
        )

    def get_collection_count(self) -> int:
        return self.vector_store._collection.count()
