from typing import List, Optional, Literal

from pydantic import BaseModel, Field
from llm.groq_client import get_groq_model


class QueryEntities(BaseModel):

    companies: List[Literal["AAPL", "MSFT", "NVDA"]] = Field(
        default_factory=list,
        description="Company tickers explicitly mentioned or clearly identified from conversation."
    )

    financial_metrics: List[str] = Field(
        default_factory=list,
        description="Financial metrics requested by the user."
    )

    fiscal_years: List[str] = Field(
        default_factory=list,
        description="Fiscal years explicitly requested, such as FY2024 or FY2025."
    )

    intent: Literal[
        "general",
        "financial_question",
        "calculation",
        "comparison"
    ] = Field(
        description="The user's primary intent."
    )


def extract_query_entities(
    question: str,
    history: str = ""
) -> QueryEntities:

    model = get_groq_model()

    structured_model = model.with_structured_output(QueryEntities)

    prompt = f"""
    You extract entities from questions for a financial research agent.

    Supported companies:
    - AAPL: Apple
    - MSFT: Microsoft
    - NVDA: NVIDIA

    Extract:
    1. Companies
    2. Financial metrics
    3. Fiscal years
    4. User intent

    Rules:
    - Use conversation history to resolve references such as "it",
      "their", and "that company".
    - Do not add companies that are not supported by the question
      or conversation.
    - Do not invent fiscal years.
    - Preserve the user's requested financial metrics.
    - Use comparison intent when two or more companies are being
      compared.
    - Use calculation intent for arithmetic or financial calculations.
    - Use financial_question for ordinary financial document questions.
    - Use general for greetings and capability questions.

    Previous conversation:
    {history}

    Current question:
    {question}
    """

    return structured_model.invoke(prompt)