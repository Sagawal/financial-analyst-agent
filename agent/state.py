from typing import TypedDict, Optional, List, Any, Dict, Annotated
from operator import or_
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    question: str
    ticker: str
    route: Optional[str]
    answer: Optional[str]
    sources: Optional[List[Any]]

    # New fields for retrieval improvement
    search_query: Optional[str]
    retrieved_documents: Optional[List[Any]]
    documents_relevant: Optional[bool]
    retry_count: int

    answer_supported: Optional[bool]
    verification_feedback: Optional[str]
    answer_retry_count: int

    comparison_tickers: Optional[List[str]]

    company_results: Annotated[Dict[str, Any], or_]

    messages: Annotated[List[Any], add_messages]

    contextual_question: Optional[str]

    companies: Optional[List[str]]
    financial_metrics: Optional[List[str]]
    fiscal_years: Optional[List[str]]
    intent: Optional[str]

    active_ticker: Optional[str]