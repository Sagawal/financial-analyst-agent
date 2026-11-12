from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm.groq_client import get_groq_model
from rag.qa_chain import retriever


COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA"
}


def analyze_company(question: str, ticker: str):

    documents = retriever.retrieve(
        query=question,
        ticker=ticker,
        k=5
    )

    if not documents:
        return {
            "company": COMPANIES[ticker],
            "answer": "No relevant documents found.",
            "sources": []
        }

    context = "\n\n".join(
        f"""
        Filename: {doc.metadata.get('filename', 'Unknown')}
        Page: {doc.metadata.get('page', 'Unknown')}
        Content: {doc.page_content}
        """
        for doc in documents
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a financial research assistant.

            Analyze the requested financial information for this company.

            Use only the supplied documents.
            Never invent financial figures.
            Mention fiscal years and units.
            If information is unavailable, say so.
            """
        ),
        (
            "human",
            """
            Company: {company}
            Question: {question}

            Financial documents:
            {context}
            """
        )
    ])

    chain = prompt | get_groq_model() | StrOutputParser()

    answer = chain.invoke({
        "company": COMPANIES[ticker],
        "question": question,
        "context": context
    })

    return {
        "company": COMPANIES[ticker],
        "answer": answer,
        "sources": documents
    }