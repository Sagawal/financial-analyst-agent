
from langchain_core.documents import Document

from rag.embeddings import BGEEmbeddings
from rag.vector_store import FinancialVectorStore
from rag.supabase_vector_store import SupabaseFinancialVectorStore


BATCH_SIZE = 100


def migrate():

    print("Initializing ChromaDB...")
    embedding_model = BGEEmbeddings()

    chroma_store = FinancialVectorStore(
        embedding_model=embedding_model
    )

    supabase_store = SupabaseFinancialVectorStore(
        embedding_model=embedding_model
    )

    collection = chroma_store.vector_store._collection

    total_documents = collection.count()

    print(f"Documents in ChromaDB: {total_documents}")

    if total_documents == 0:
        print("No documents to migrate.")
        return

    migrated = 0

    for offset in range(0, total_documents, BATCH_SIZE):

        batch = collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas"]
        )

        documents = []

        for content, metadata in zip(
            batch["documents"],
            batch["metadatas"]
        ):
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata or {}
                )
            )

        if documents:
            supabase_store.add_documents(documents)

            migrated += len(documents)

        print(
            f"Migrated {migrated}/{total_documents} documents"
        )

    print("\nMigration completed!")
    print("ChromaDB documents:", total_documents)
    print("Documents processed:", migrated)
    print(
        "Supabase total records:",
        supabase_store.get_collection_count()
    )


if __name__ == "__main__":
    migrate()
