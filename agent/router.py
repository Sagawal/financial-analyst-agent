from typing import Literal
from pydantic import BaseModel, Field

from llm.groq_client import get_groq_model


class RouteDecision(BaseModel):
    route: Literal[
        "general",
        "rag",
        "calculator",
        "financial_analysis",
        "comparison",
        "market_data",
        "web_search"
    ] = Field(
        description="The best workflow route for the user's question."
    )


def get_query_route(question: str, history: str = "") -> str:

    model = get_groq_model()

    structured_model = model.with_structured_output(RouteDecision)

    
    prompt = f"""
        You are a query classifier for a Financial Analyst Agent.

        Previous conversation:
        {history}

        Current user question:
        {question}

        Classify the CURRENT question into exactly one route:

        1. general:
        Greetings, introductions, and questions about the assistant's
        capabilities.

        2. rag:
        Questions asking to retrieve, explain, summarize, or identify
        information from financial documents.

        Examples:
        - What was Apple's annual revenue?
        - What are NVIDIA's business segments?
        - What was Microsoft's operating income?

        3. calculator:
        Basic arithmetic or calculations where the user provides
        all required numerical values.

        Examples:
        - Calculate 20% of 500.
        - Calculate percentage growth from 100 to 120.

        4. financial_analysis:
        Questions involving financial metrics, ratios, growth rates,
        or calculations that require retrieving financial figures
        from company documents.

        Examples:
        - What was Microsoft's net profit margin?
        - Calculate Apple's revenue growth over two years.
        - What is NVIDIA's debt-to-equity ratio?

        5. comparison:
        Questions comparing financial information of two or more
        supported companies using their financial documents.

        Examples:
        - Compare Apple and NVIDIA's annual revenue.
        - Compare profitability of all three companies.

        6. market_data:
        Questions about current or historical stock market information.

        Examples:
        - What is Apple's current stock price?
        - Show NVIDIA's stock price history for the last month.
        - What is Microsoft's market capitalization?
        - Compare Apple's and NVIDIA's current stock prices.

        Use market_data for stock prices and market information,
        not for revenue or profit figures from annual reports.

        7. web_search:
        Questions about recent financial news, company announcements,
        earnings news, acquisitions, regulatory developments, and
        current economic events.

        Examples:
        - What are the latest news about NVIDIA?
        - Has Apple announced any new products recently?
        - What are the latest developments affecting Microsoft?
        - What recent regulatory actions affect NVIDIA?

        Use web_search for current news and developments.
        Use market_data for stock prices and market statistics.
        Use rag for historical information from uploaded financial documents.

        Routing priorities:
        - Recent financial news and current company developments -> web_search.
        - Current or historical stock market information -> market_data.
        - Comparison of financial information across companies -> comparison.
        - Financial calculations requiring values from documents -> financial_analysis.
        - Calculations using values already supplied by the user -> calculator.
        - Other financial document questions -> rag.
        - Greetings and capability questions -> general.
        
        

        Conversation handling:
        - Use conversation history to understand references such as
        "their", "it", "that company", and "what about the previous year?"
        - Classify based primarily on the current question and its context.
        - Do not classify based on previous questions alone.
        - If a follow-up question continues a previous task, preserve
        the relevant intent and context.
        - If uncertain, use rag.

        Return exactly one route.
        """


    try:
        decision = structured_model.invoke(prompt)
        return decision.route

    except Exception:
        # Safe fallback if structured output fails
        return "rag"