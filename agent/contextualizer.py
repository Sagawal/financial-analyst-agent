from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm.groq_client import get_groq_model


def contextualize_question(question: str, history: str) -> str:

    if not history.strip():
        return question

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a contextual question rewriter for a financial
            research assistant.

            Rewrite the current user question into a standalone
            question using the previous conversation when necessary.

            Rules:
            1. Resolve references such as "it", "they", "their",
               "that company", and "what about 2023".
            2. Preserve the user's original intent.
            3. Preserve company names, financial metrics and fiscal years.
            4. Do not introduce facts or assumptions.
            5. If the current question is already standalone,
               return it unchanged.
            6. Return only the rewritten question.
            """
        ),
        (
            "human",
            """
            Previous conversation:
            {history}

            Current question:
            {question}
            """
        )
    ])

    chain = prompt | get_groq_model() | StrOutputParser()

    result = chain.invoke({
        "history": history,
        "question": question
    })

    return result.strip()