
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm.groq_client import get_groq_model
from rag.embeddings import BGEEmbeddings
from rag.vector_store import FinancialVectorStore
from rag.retriever import FinancialRetriever

import os

from rag.supabase_vector_store import SupabaseFinancialVectorStore


# 1. Initialize components
embedding_model = BGEEmbeddings()

VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE", "chroma").lower()

if VECTOR_STORE_TYPE == "chroma":

    vector_store = FinancialVectorStore(
        embedding_model=embedding_model
    )

elif VECTOR_STORE_TYPE == "supabase":

    vector_store = SupabaseFinancialVectorStore(
        embedding_model=embedding_model
    )

else:
    raise ValueError(
        f"Unsupported vector store: {VECTOR_STORE_TYPE}"
    )

retriever = FinancialRetriever(vector_store)

llm = get_groq_model()


# 2. Define prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a professional financial research assistant.

        Answer questions using ONLY the provided context.

        Rules:
        1. Do not invent financial figures.
        2. If information is unavailable, clearly say so.
        3. Explain financial concepts in simple language.
        4. Cite sources using their source identifiers.
        5. Distinguish reported facts from your interpretation.
        6. Mention the fiscal year whenever relevant.
        7. Do not provide personalized investment advice.

        Context:
        {context}
        """
    ),
    (
        "human",
        "{question}"
    )
])


# 3. Format retrieved documents
def format_documents(documents):

    formatted_documents = []

    for index, document in enumerate(documents, start=1):

        metadata = document.metadata

        source = metadata.get("filename", "Unknown source")
        ticker = metadata.get("ticker", "Unknown")
        page = metadata.get("page", "Not available")
        section = metadata.get("section", "Not available")
        filing_type = metadata.get("filing_type", "Unknown")

        formatted_documents.append(
            f"""
            Source ID: {index}
            Company: {ticker}
            Filing: {filing_type}
            Section: {section}
            File: {source}
            Page: {page}

            Content:
            {document.page_content}
            """
        )

    return "\n\n".join(formatted_documents)


# 4. Create RAG function
def ask_question(
    question: str,
    ticker: str | None = None,
    filing_type: str | None = None,
    k: int = 5
):

    # Retrieve relevant chunks
    documents = retriever.retrieve(
        query=question,
        ticker=ticker,
        filing_type=filing_type,
        k=k
    )

    if not documents:
        return {
            "answer": "I couldn't find relevant information in the indexed documents.",
            "sources": []
        }

    # Prepare context
    context = format_documents(documents)

    # Construct chain
    rag_chain = prompt | llm | StrOutputParser()

    # Generate answer
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": documents
    }
