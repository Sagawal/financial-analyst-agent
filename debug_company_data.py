from rag.embeddings import BGEEmbeddings
from rag.vector_store import FinancialVectorStore

embedding_model = BGEEmbeddings()
vector_store = FinancialVectorStore(
    embedding_model=embedding_model
)

for ticker in ["AAPL", "MSFT", "NVDA"]:

    print(f"\n{'=' * 50}")
    print(f"TICKER: {ticker}")
    print("=" * 50)

    documents = vector_store.similarity_search(
        query="annual revenue total revenue",
        k=10,
        filters={"ticker": ticker}
    )

    for doc in documents:
        print("\nTicker:", doc.metadata.get("ticker"))
        print("Company:", doc.metadata.get("company"))
        print("Filename:", doc.metadata.get("filename"))
        print("Source:", doc.metadata.get("source"))
        print("Page:", doc.metadata.get("page"))
        print("Content:", doc.page_content[:250])