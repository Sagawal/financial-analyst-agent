from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from agent.checkpointer import checkpointer
from langchain_tavily import TavilySearch

from agent.state import AgentState
from agent.router import get_query_route

from agent.retrieval import grade_documents, rewrite_query
from langchain_core.output_parsers import StrOutputParser
from agent.verification import verify_answer
from agent.memory import get_conversation_history

from rag.qa_chain import retriever
from llm.groq_client import get_groq_model

from agent.contextualizer import contextualize_question
from agent.memory import get_conversation_history

from tools.market_data import (
    get_stock_price,
    get_historical_stock_data,
    get_company_market_info
)

from agent.entities import extract_query_entities

from langgraph.types import Send
from agent.comparison import analyze_company

from agent.tools import (
    calculator,
    revenue_growth,
    net_profit_margin,
    debt_to_equity
)


# ---------------- Router Node ----------------

def router_node(state: AgentState):

    question = (
        state.get("contextual_question")
        or state["question"]
    )

    history = get_conversation_history(
        state,
        max_messages=10,
        exclude_last=True
    )

    route = get_query_route(
        question=question,
        history=history
    )

    companies = state.get("companies") or []

    # Use extracted companies for comparisons
    if len(companies) >= 2:
        route = "comparison"

    # Use the explicitly identified company for single-company queries
    elif len(companies) == 1:
        route_ticker = companies[0]
    else:
        route_ticker = state["ticker"]

    if route != "comparison":
        if len(companies) == 1:
            route_ticker = companies[0]
        else:
            route_ticker = state["ticker"]
    else:
        route_ticker = None

    return {
        "route": route,
        "comparison_tickers": companies if route == "comparison" else [],
        "active_ticker": route_ticker
    }


# ---------------- Retrieve Node ----------------

def retrieve_node(state: AgentState):

    search_query = state.get("contextual_question") or state["question"]

    documents = retriever.retrieve(
        query=search_query,
        ticker=state.get("active_ticker") or state["ticker"],
        k=5
    )

    return {
        "retrieved_documents": documents,
        "sources": documents
    }


# ---------------- Relevance Grader Node ----------------

def grade_node(state: AgentState):

    documents = state.get("retrieved_documents", [])

    is_relevant = grade_documents(
        question=state["question"],
        documents=documents
    )

    return {"documents_relevant": is_relevant}


# ---------------- Query Rewrite Node ----------------

def rewrite_node(state: AgentState):

    current_query = (
        state.get("search_query") or state["question"]
    )

    rewritten_question = rewrite_query(current_query)

    return {
        "search_query": rewritten_question,
        "retry_count": state.get("retry_count", 0) + 1
    }


# ---------------- Retrieval Routing ----------------

def check_relevance(state: AgentState):

    if state["documents_relevant"]:
        return "generate"

    if state.get("retry_count", 0) < 2:
        return "rewrite"

    return "generate"

# ---------------- Answer Generation Node ----------------

def generate_answer_node(state: AgentState):

    if not state.get("documents_relevant"):
        return {
            "answer": (
                "I couldn't find sufficiently relevant information "
                "in the available financial documents, even after "
                "retrying retrieval. Please try rephrasing your question "
                "or upload a relevant financial report."
            )
        }

    documents = state.get("retrieved_documents", [])

    if not documents:
        return {
            "answer": "I couldn't find relevant information in the available financial documents."
        }

    context_parts = []

    for i, doc in enumerate(documents, start=1):
        metadata = doc.metadata

        source_info = [
            f"Source ID: {i}",
            f"Filename: {metadata.get('filename', 'Unknown')}",
            f"Document type: {metadata.get('document_type', 'Unknown')}",
        ]

        if metadata.get("document_type") == "sec_filing":
            source_info.extend([
                f"Filing type: {metadata.get('filing_type', 'Unknown')}",
                f"Filing date: {metadata.get('filing_date', 'Unknown')}",
                f"Report date: {metadata.get('report_date', 'Unknown')}",
                f"Section: {metadata.get('section', 'Unknown')}",
            ])

        else:
            source_info.append(
                f"Page: {metadata.get('page', 'Unknown')}"
            )

        context_parts.append(
            "\n".join(source_info) +
            f"\nContent: {doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    verification_feedback = state.get("verification_feedback")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a financial research assistant.

            Answer the user's question using ONLY the retrieved context.

            Rules:
            - Answer using ONLY the retrieved context.
            - Never invent financial values.
            - Mention fiscal years and units.
            - Distinguish facts from interpretation.
            - If information is missing, say so.
            - Cite factual claims using [Source 1], [Source 2], etc.
            - Use only source IDs provided in the retrieved context.
            - Place citations next to the claims they support.
            - Never invent page numbers, filing dates, or sections.

            Additional verification feedback:
            {feedback}

            Retrieved context:
            {context}
            """
        ),
        ("human", "{question}")
    ])

    chain = prompt | get_groq_model() | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": state["question"],
        "feedback": verification_feedback or "No previous verification feedback."
    })

    return {"answer": answer}


# ---------------- Calculator Node ----------------

def calculator_node(state: AgentState):

    model = get_groq_model()

    financial_tools = [
        calculator,
        revenue_growth,
        net_profit_margin,
        debt_to_equity
    ]

    model_with_tools = model.bind_tools(financial_tools)

    tools_by_name = {
        tool.name: tool for tool in financial_tools
    }

    messages = [
        SystemMessage(
        content="""
        You are a financial analysis assistant.

        Use the available tools for calculations.

        Available tools:
        - calculator: basic arithmetic
        - revenue_growth: revenue growth percentage
        - net_profit_margin: net profit margin
        - debt_to_equity: debt-to-equity ratio

        Rules:
        1. Never invent financial values.
        2. If required values are missing, ask the user for them.
        3. Include the formula and explain the result.
        4. Mention the units supplied by the user.
        5. Do not interpret a ratio as universally good or bad;
        explain that interpretation depends on the industry.
        """
    ),
        HumanMessage(content=state["question"])
    ]

    response = model_with_tools.invoke(messages)

    if response.tool_calls:
        messages.append(response)

        for tool_call in response.tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]

            result = selected_tool.invoke(tool_call["args"])

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

        final_response = model_with_tools.invoke(messages)
    else:
        final_response = response



    return {
        "answer": final_response.content,
        "sources": []
    }


# ---------------- Financial Analysis Node ----------------

def financial_analysis_node(state: AgentState):

    # 1. Retrieve relevant financial documents
    documents = retriever.retrieve(
        query=state["question"],
        ticker=state.get("active_ticker") or state["ticker"],
        k=10
    )

    if not documents:
        return {
            "answer": "I couldn't find relevant financial information in the available documents.",
            "sources": []
        }

    # 2. Prepare retrieved context
    context_parts = []

    for i, doc in enumerate(documents, start=1):
        metadata = doc.metadata

        context_parts.append(
            f"""
            Source {i}
            Filename: {metadata.get('filename', 'Unknown')}
            Page: {metadata.get('page', 'Unknown')}
            Filing type: {metadata.get('filing_type', 'Unknown')}

            Content:
            {doc.page_content}
            """
        )

    context = "\n\n".join(context_parts)

    # 3. Bind financial tools
    financial_tools = [
        calculator,
        revenue_growth,
        net_profit_margin,
        debt_to_equity
    ]

    model = get_groq_model()
    model_with_tools = model.bind_tools(financial_tools)

    tools_by_name = {
        tool.name: tool for tool in financial_tools
    }

    messages = [
        SystemMessage(
            content=f"""
            You are a financial research and analysis assistant.

            Answer the user's question using ONLY the retrieved context.

            Retrieved financial documents:
            {context}

            Instructions:
            1. Extract financial values only when explicitly supported
               by the retrieved documents.
            2. Never invent or assume missing financial values.
            3. Use the appropriate financial tool for calculations.
            4. Do not calculate a metric if its required inputs are missing.
            5. Pay attention to fiscal years and units.
            6. Clearly distinguish retrieved facts from calculated results.
            7. Cite source filenames and page numbers in your answer.
            8. If the documents do not contain sufficient information,
               explain what is missing.
            """
        ),
        HumanMessage(content=state["question"])
    ]

    # 4. Ask LLM to analyze and call tools if needed
    response = model_with_tools.invoke(messages)

    if response.tool_calls:
        messages.append(response)

        for tool_call in response.tool_calls:

            selected_tool = tools_by_name[tool_call["name"]]

            result = selected_tool.invoke(tool_call["args"])

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

        final_response = model_with_tools.invoke(messages)

    else:
        final_response = response

    return {
        "answer": final_response.content,
        "sources": documents
    }

# ---------------- General Node ----------------

def general_node(state: AgentState):

    return {
        "answer": (
            "Hello! I'm your Financial Analyst Agent. "
            "I can answer questions about company financial filings "
            "and perform financial calculations."
        ),
        "sources": []
    }

# ---------------- Answer Verification Node ----------------

def verify_node(state: AgentState):

    documents = state.get("retrieved_documents", [])

    result = verify_answer(
        question=state["question"],
        answer=state["answer"],
        documents=documents
    )

    return {
        "answer_supported": result.supported,
        "verification_feedback": result.feedback
    }


# ---------------- Answer Revision Node ----------------

def revise_answer_node(state: AgentState):

    return {
        "answer_retry_count": state.get("answer_retry_count", 0) + 1
    }


# ---------------- Contextualization Node ----------------

def contextualize_node(state: AgentState):

    history = get_conversation_history(
        state,
        max_messages=10,
        exclude_last=True
    )

    contextual_question = contextualize_question(
        question=state["question"],
        history=history
    )

    return {
        "contextual_question": contextual_question,
        "search_query": contextual_question
    }


# ---------------- Entity Extraction Node ----------------

def extract_entities_node(state: AgentState):

    history = get_conversation_history(
        state,
        max_messages=10,
        exclude_last=True
    )

    entities = extract_query_entities(
        question=state.get("contextual_question") or state["question"],
        history=history
    )

    return {
        "companies": entities.companies,
        "financial_metrics": entities.financial_metrics,
        "fiscal_years": entities.fiscal_years,
        "intent": entities.intent,
        "comparison_tickers": (
            entities.companies
            if entities.intent == "comparison"
            else []
        )
    }


# ---------------- Market Data Node ----------------

def market_data_node(state: AgentState):

    ticker = (
        state.get("active_ticker")
        or state.get("ticker")
    )

    financial_tools = [
        get_stock_price,
        get_historical_stock_data,
        get_company_market_info
    ]

    tools_by_name = {
        tool.name: tool for tool in financial_tools
    }

    model = get_groq_model()
    model_with_tools = model.bind_tools(financial_tools)

    messages = [
        SystemMessage(
            content=f"""
            You are a financial market data assistant.

            The selected company ticker is {ticker}.

            Use the available tools to retrieve market information.

            Rules:
            1. Always use a tool for stock prices and market information.
            2. Use the selected ticker unless the user explicitly
               identifies another supported company.
            3. Never invent stock prices or market capitalization.
            4. Clearly mention the ticker and currency.
            5. Explain that Yahoo Finance data may be delayed or
               incomplete and is not an official exchange feed.
            6. Do not provide personalized investment advice.
            """
        ),
        HumanMessage(
            content=state.get("contextual_question") or state["question"]
        )
    ]

    response = model_with_tools.invoke(messages)

    if not response.tool_calls:
        return {
            "answer": response.content,
            "sources": []
        }

    messages.append(response)

    for tool_call in response.tool_calls:

        selected_tool = tools_by_name[tool_call["name"]]

        result = selected_tool.invoke(tool_call["args"])

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    final_response = model_with_tools.invoke(messages)

    return {
        "answer": final_response.content,
        "sources": []
    }


# ---------------- Web Search Node ----------------

def web_search_node(state: AgentState):

    question = (
        state.get("contextual_question")
        or state["question"]
    )

    search_tool = TavilySearch(
        max_results=5,
        topic="finance",
        search_depth="basic"
    )

    search_results = search_tool.invoke({
        "query": question
    })

    results = search_results.get("results", [])

    if not results:
        return {
            "answer": "No relevant recent financial news was found.",
            "sources": []
        }

    context_parts = []

    for i, item in enumerate(results, start=1):
        context_parts.append(
            f"""
            Source {i}
            Title: {item.get('title', 'Unknown')}
            URL: {item.get('url', 'Unavailable')}
            Published: {item.get('published_date', 'Unknown')}
            Content: {item.get('content', '')}
            """
        )

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a financial news research assistant.

            Answer the user's question using ONLY the web search results.

            Rules:
            1. Summarize the most relevant developments.
            2. Do not invent facts, dates, or figures.
            3. Distinguish reported facts from analysis.
            4. Mention publication dates when available.
            5. Cite claims using clickable source links in Markdown.
            6. If sources disagree, mention the disagreement.
            7. If the results are insufficient, say so.

            Web search results:
            {context}
            """
        ),
        ("human", "{question}")
    ])

    chain = prompt | get_groq_model() | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": []
    }

# ---------------- Company Detection ----------------

def detect_comparison_companies(question: str):

    text = question.lower()

    aliases = {
        "AAPL": ["apple", "aapl"],
        "MSFT": ["microsoft", "msft"],
        "NVDA": ["nvidia", "nvda"]
    }

    tickers = []

    for ticker, names in aliases.items():
        if any(name in text for name in names):
            tickers.append(ticker)

    return tickers


# ---------------- Comparison Dispatcher ----------------

def dispatch_comparison(state: AgentState):

    tickers = state.get("comparison_tickers") or []

    return [
        Send(
            "company_analysis",
            {
                "question": state["question"],
                "ticker": ticker
            }
        )
        for ticker in tickers
    ]


# ---------------- Parallel Company Analysis ----------------

def company_analysis_node(state: AgentState):

    ticker = state["ticker"]

    result = analyze_company(
        question=state["question"],
        ticker=ticker
    )

    return {
        "company_results": {
            ticker: result
        }
    }


# ---------------- Comparison Synthesis ----------------

def comparison_node(state: AgentState):

    results = state.get("company_results", {})

    if not results:
        return {
            "answer": "No company comparison results are available.",
            "sources": []
        }

    comparison_context = []
    all_sources = []

    for ticker, result in results.items():

        comparison_context.append(
            f"""
            Company: {result['company']}
            Ticker: {ticker}
            Analysis: {result['answer']}
            """
        )

        all_sources.extend(result["sources"])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a financial comparison assistant.

            Compare companies using only the supplied company analyses.

            Rules:
            - Do not invent missing figures.
            - Do not assume different fiscal years are directly comparable.
            - Preserve the units of financial values.
            - Clearly identify missing information.
            - Explain the comparison in simple language.
            - Do not provide investment advice.
            """
        ),
        (
            "human",
            """
            Original question:
            {question}

            Company analyses:
            {context}

            Prepare a structured comparison.
            """
        )
    ])

    chain = prompt | get_groq_model() | StrOutputParser()

    answer = chain.invoke({
        "question": state["question"],
        "context": "\n\n".join(comparison_context)
    })

    return {
        "answer": answer,
        "sources": all_sources
    }


# ---------------- Verification Routing ----------------

def verification_route(state: AgentState):

    if state.get("answer_supported"):
        return "finish"

    if state.get("answer_retry_count", 0) < 1:
        return "revise"

    return "finish"


# ---------------- Conditional Routing ----------------

def route_query(state: AgentState):
    return state["route"]


# ---------------- Build Graph ----------------

workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("router", router_node)
workflow.add_node("calculator", calculator_node)
workflow.add_node("financial_analysis", financial_analysis_node)
workflow.add_node("general", general_node)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_answer_node)

workflow.add_node("verify", verify_node)
workflow.add_node("revise", revise_answer_node)

workflow.add_node(
    "comparison_dispatch",
    lambda state: {}
)

workflow.add_node("company_analysis", company_analysis_node)

workflow.add_node(
    "comparison",
    comparison_node
)

workflow.add_node("contextualize", contextualize_node)
workflow.add_node("extract_entities", extract_entities_node)
workflow.add_node("market_data", market_data_node)
workflow.add_node("web_search", web_search_node)


# Entry point

workflow.add_edge(START, "contextualize")
workflow.add_edge("contextualize", "extract_entities")
workflow.add_edge("extract_entities", "router")


# Initial routing
workflow.add_conditional_edges(
    "router",
    route_query,
    {
        "rag": "retrieve",
        "calculator": "calculator",
        "financial_analysis": "financial_analysis",
        "general": "general",
        "comparison": "comparison_dispatch",
        "market_data": "market_data",
        "web_search": "web_search"
    }
)

workflow.add_conditional_edges(
    "comparison_dispatch",
    dispatch_comparison
)

workflow.add_conditional_edges(
    "grade",
    check_relevance,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

workflow.add_conditional_edges(
    "verify",
    verification_route,
    {
        "revise": "revise",
        "finish": END
    }
)

workflow.add_edge("company_analysis", "comparison")
workflow.add_edge("comparison", END)

# RAG retrieval workflow
workflow.add_edge("retrieve", "grade")



workflow.add_edge("rewrite", "retrieve")


# Answer verification workflow
# workflow.add_edge("generate", "verify")
workflow.add_edge("generate", "verify")



workflow.add_edge("revise", "generate")
workflow.add_edge("market_data", END)
workflow.add_edge("web_search", END)


# Other routes
workflow.add_edge("calculator", END)
workflow.add_edge("financial_analysis", END)
workflow.add_edge("general", END)


# Compile
agent_graph = workflow.compile(
    checkpointer=checkpointer
)


# ---------------- Public Function ----------------

def ask_agent(question: str, ticker: str, thread_id: str):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    initial_state = {
        "question": question,
        "ticker": ticker,
        "route": None,
        "answer": None,
        "sources": [],
        "search_query": None,
        "retrieved_documents": [],
        "documents_relevant": None,
        "retry_count": 0,
        "answer_supported": None,
        "verification_feedback": None,
        "answer_retry_count": 0,
        "comparison_tickers": [],
        "company_results": {},
        "messages": [HumanMessage(content=question)],
        "contextual_question": None,
        "companies": [],
        "financial_metrics": [],
        "fiscal_years": [],
        "intent": None,
        "active_ticker": None,
    }

    result = agent_graph.invoke(
        initial_state,
        config=config
    )

    answer = result.get("answer", "")

    # Save assistant response in conversation memory
    agent_graph.update_state(
        config,
        {
            "messages": [
                AIMessage(content=answer)
            ]
        }
    )

    return {
        "answer": answer,
        "sources": result.get("sources", []),
        "route": result.get("route"),
        "answer_supported": result.get("answer_supported")
    }