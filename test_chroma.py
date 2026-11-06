
from langchain_core.documents import Document

from rag.embeddings import BGEEmbeddings
from rag.vector_store import FinancialVectorStore
from rag.retriever import FinancialRetriever


# 1. Initialize embedding model
embedding_model = BGEEmbeddings()

# 2. Initialize vector database
vector_store = FinancialVectorStore(
    embedding_model=embedding_model
)

# 3. Prepare sample documents
documents = [
    Document(
        page_content="Apple reported annual revenue of 391 billion dollars.",
        metadata={
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "filing_type": "10-K",
            "fiscal_year": 2024,
            "page": 15,
            "chunk_index": 0
        }
    ),
    Document(
        page_content="Microsoft reported strong growth in its cloud business.",
        metadata={
            "ticker": "MSFT",
            "company": "Microsoft Corporation",
            "filing_type": "10-K",
            "fiscal_year": 2024,
            "page": 20,
            "chunk_index": 0
        }
    ),
    Document(
        page_content="NVIDIA reported increased demand for AI GPUs.",
        metadata={
            "ticker": "NVDA",
            "company": "NVIDIA Corporation",
            "filing_type": "10-K",
            "fiscal_year": 2024,
            "page": 12,
            "chunk_index": 0
        }
    )
]

# 4. Store documents
ids = vector_store.add_documents(documents)

print("Inserted IDs:", ids)
print("Collection count:", vector_store.get_collection_count())

# 5. Initialize retriever
retriever = FinancialRetriever(vector_store)

# 6. Retrieve relevant documents
results = retriever.retrieve(
    query="What was Apple's revenue?",
    ticker="AAPL",
    k=2
)

# 7. Display results
for result in results:
    print("\nContent:", result.page_content)
    print("Metadata:", result.metadata)
