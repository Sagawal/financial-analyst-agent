from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from llm.groq_client import get_groq_model


class VerificationDecision(BaseModel):
    supported: bool = Field(
        description="Whether all factual claims in the answer are supported by the provided context."
    )

    feedback: str = Field(
        description="Explain which claims are unsupported or what information is missing."
    )


def verify_answer(question: str, answer: str, documents: list):

    if not documents:
        return VerificationDecision(
            supported=False,
            feedback="No source documents were provided."
        )

    context = "\n\n".join(
        f"""
        Source: {doc.metadata.get('filename', 'Unknown')}
        Page: {doc.metadata.get('page', 'Unknown')}
        Content: {doc.page_content}
        """
        for doc in documents
    )

    model = get_groq_model()
    verifier = model.with_structured_output(VerificationDecision)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a financial answer verification assistant.

            Your task is to verify whether the answer is supported
            by the provided source documents.

            Rules:
            1. Check every factual claim and financial figure.
            2. Check fiscal years and numerical units.
            3. Do not use outside knowledge.
            4. If a claim is not supported, mark supported=false.
            5. If the answer makes unsupported assumptions,
               mark supported=false.
            6. If the answer accurately communicates that information
               is unavailable, it can be supported=true.
            7. Explain unsupported claims in the feedback.
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Generated Answer:
            {answer}

            Source Documents:
            {context}
            """
        )
    ])

    chain = prompt | verifier

    return chain.invoke({
        "question": question,
        "answer": answer,
        "context": context
    })