from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from llm.groq_client import get_groq_model


class RelevanceDecision(BaseModel):
    relevant: bool = Field(
        description="Whether the documents contain information useful for answering the question."
    )


def grade_documents(question: str, documents: list) -> bool:

    if not documents:
        return False

    model = get_groq_model()
    grader = model.with_structured_output(RelevanceDecision)

    context = "\n\n".join(
        doc.page_content[:1500] for doc in documents[:5]
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a relevance grader.

            Determine whether the retrieved financial documents
            contain information useful for answering the question.

            Return relevant=true if the documents contain directly
            useful evidence or information.

            Return relevant=false if they are unrelated or do not
            contain useful information.
            """
        ),
        (
            "human",
            "Question:\n{question}\n\nRetrieved documents:\n{context}"
        )
    ])

    chain = prompt | grader

    result = chain.invoke({
        "question": question,
        "context": context
    })

    return result.relevant


def rewrite_query(question: str) -> str:

    model = get_groq_model()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a financial search query optimizer.

            Rewrite the user's question to improve retrieval from
            financial reports and SEC filings.

            Preserve the original intent, company, financial metric,
            and fiscal-year constraints.

            Return only the rewritten search query.
            Do not answer the question.
            """
        ),
        ("human", "{question}")
    ])

    chain = prompt | model

    result = chain.invoke({"question": question})

    return result.content.strip()