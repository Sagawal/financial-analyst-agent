
from langchain_core.documents import Document

from rag.embeddings import BGEEmbeddings
from rag.vector_store import FinancialVectorStore


class FinancialRetriever:

    def __init__(self, vector_store: FinancialVectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        ticker: str | None = None,
        filing_type: str | None = None,
        k: int = 5
    ) -> list[Document]:

        filters = {}

        if ticker:
            filters["ticker"] = ticker

        if filing_type:
            filters["filing_type"] = filing_type

        # Chroma uses $and when multiple conditions are required.
        if len(filters) > 1:
            filters = {
                "$and": [
                    {key: {"$eq": value}}
                    for key, value in filters.items()
                ]
            }

        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filters=filters or None
        )
