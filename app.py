
import os
import uuid
import json

import streamlit as st
import pandas as pd


# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="Financial Analyst Agent",
    page_icon="📊",
    layout="wide"
)


# ---------------- Custom Styling ----------------


# ---------------- Custom Styling ----------------

st.markdown("""
<style>
    /* Main application background */
    .stApp {
        background-color: #0B1020;
        color: #E5E7EB;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #151B2D;
        border-right: 1px solid #252D42;
    }

    /* Headings */
    h1, h2, h3 {
        color: #F8FAFC !important;
        letter-spacing: -0.5px;
    }

    /* Normal text */
    p, label, span, li {
        color: #D1D5DB;
    }

    .app-subtitle {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-top: -12px;
        margin-bottom: 25px;
    }

    /* Navigation tabs */
    button[data-baseweb="tab"] {
        color: #C4B5FD !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #A78BFA !important;
        border-bottom: 3px solid #7047EB !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #7047EB;
        color: #FFFFFF !important;
        border: 1px solid #805CF0;
        border-radius: 8px;
    }

    .stButton > button:hover {
        background-color: #805CF0;
        border-color: #A78BFA;
    }

    /* Input fields */
    input, textarea {
        background-color: #151B2D !important;
        color: #F8FAFC !important;
        border-color: #374151 !important;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #151B2D;
        border-color: #374151;
        color: #F8FAFC;
    }

    /* Expanders and containers */
    div[data-testid="stExpander"] {
        background-color: #151B2D;
        border: 1px solid #29334A;
        border-radius: 10px;
    }

    div[data-testid="stMetric"] {
        background-color: #151B2D;
        border: 1px solid #7047EB;
        padding: 15px;
        border-radius: 10px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #151B2D;
        border: 1px solid #29334A;
        border-radius: 12px;
        padding: 12px;
    }

    /* Dividers */
    hr {
        border-color: #29334A;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #29334A;
        border-radius: 8px;
    }

    /* Links */
    a {
        color: #A78BFA !important;
    }

</style>
""", unsafe_allow_html=True)



# ---------------- Load Streamlit Cloud Secrets ----------------

try:
    secrets = st.secrets

    secret_keys = [
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "TAVILY_API_KEY",
        "SEC_USER_AGENT",
        "EMBEDDING_MODEL",
        "SUPABASE_DB_URL",
        "VECTOR_STORE",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "LANGGRAPH_DATABASE_URL",
    ]

    for key in secret_keys:
        if key in secrets and secrets[key]:
            os.environ.setdefault(key, str(secrets[key]))

except Exception:
    pass


# ---------------- Application Imports ----------------

from agent.graph import ask_agent
from rag.qa_chain import vector_store
from ingestion.upload_pipeline import ingest_pdf_bytes
from utils.document_manager import list_uploaded_documents
from tools.market_data import get_historical_stock_data


# ---------------- Header ----------------

st.markdown("""
# 📊 Financial Analyst Agent
<p class="app-subtitle">
    AI-powered financial research using RAG, LangGraph and financial tools.
</p>
""", unsafe_allow_html=True)


# ---------------- Conversation ID ----------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- Sidebar ----------------

st.sidebar.title("Company Context")

companies = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA"
}

selected_company = st.sidebar.selectbox(
    "Select Company",
    options=list(companies.keys())
)

ticker = companies[selected_company]

st.sidebar.info(f"Selected company: {ticker}")

st.sidebar.divider()

st.sidebar.subheader("Quick Actions")

if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()

st.sidebar.divider()

st.sidebar.caption("Financial Analyst Agent")
st.sidebar.caption("Powered by LangGraph and Supabase")


# ---------------- Source Display ----------------

def display_sources(sources):

    with st.expander("📚 View Retrieved Sources"):

        for i, doc in enumerate(sources, start=1):

            metadata = doc.metadata

            st.markdown(f"**Source {i}**")
            st.write(f"Ticker: {metadata.get('ticker', 'N/A')}")
            st.write(f"File: {metadata.get('filename', 'N/A')}")

            if metadata.get("document_type") == "sec_filing":
                st.write(f"Filing type: {metadata.get('filing_type', 'N/A')}")
                st.write(f"Filing date: {metadata.get('filing_date', 'N/A')}")
                st.write(f"Report date: {metadata.get('report_date', 'N/A')}")
                st.write(f"Section: {metadata.get('section', 'N/A')}")
            else:
                st.write(f"Page: {metadata.get('page', 'N/A')}")

            st.text(doc.page_content[:1000])
            st.divider()


# ---------------- Navigation ----------------

selected_tab = st.radio(
    "Navigation",
    ["💬 Chat", "📁 Documents", "📈 Analysis"],
    horizontal=True,
    label_visibility="collapsed"
)


# =========================================================
# CHAT TAB
# =========================================================

if selected_tab == "💬 Chat":

    st.subheader("Financial Research Assistant")
    st.caption(
        f"Ask questions about {selected_company}'s financial performance, "
        "filings and market information."
    )

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                display_sources(message["sources"])

    question = st.chat_input("Ask a financial question...")

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Analyzing financial information..."):

                try:

                    result = ask_agent(
                        question=question,
                        ticker=ticker,
                        thread_id=st.session_state.thread_id
                    )

                    if result.get("answer_supported") is True:
                        st.success("Answer verified against retrieved sources.")

                    elif result.get("answer_supported") is False:
                        st.warning(
                            "Some claims could not be verified against "
                            "the retrieved sources."
                        )

                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        display_sources(sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Something went wrong: {e}")


# =========================================================
# DOCUMENTS TAB
# =========================================================

elif selected_tab == "📁 Documents":

    st.subheader("Financial Documents")
    st.caption("Upload and manage financial reports for the selected company.")

    uploaded_files = st.file_uploader(
        "Upload financial reports (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "📥 Ingest Documents",
        disabled=not uploaded_files
    ):

        try:

            with st.spinner("Extracting, chunking and indexing PDFs..."):

                for uploaded_file in uploaded_files:

                    result = ingest_pdf_bytes(
                        file_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        ticker=ticker,
                        company=selected_company,
                        vector_store=vector_store
                    )

                    st.success(
                        f"{result['filename']}: "
                        f"{result['chunks_created']} chunks indexed."
                    )

            st.success("Document ingestion completed!")

        except Exception as e:
            st.error(f"Ingestion failed: {e}")

    st.divider()

    st.subheader(f"Uploaded Documents — {ticker}")

    documents = list_uploaded_documents(ticker)

    if not documents:

        st.info("No PDF documents uploaded for this company yet.")

    else:

        st.metric("Total Documents", len(documents))

        for document in documents:

            with st.container(border=True):

                st.markdown(f"📄 **{document['filename']}**")

                col1, col2 = st.columns(2)

                col1.caption(f"Size: {document['size_kb']} KB")
                col2.caption(f"Modified: {document['uploaded_at']}")


# =========================================================
# ANALYSIS TAB
# =========================================================

elif selected_tab == "📈 Analysis":
        
    st.subheader("Stock Market Analysis")

    st.caption(
        f"Historical market data for {selected_company} ({ticker})"
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        selected_period = st.selectbox(
            "Historical Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=0
        )

    with col2:

        st.write("")
        st.write("")
        st.caption("Select a period and generate the historical stock chart.")

    if st.button("📊 Generate Stock Chart"):

        with st.spinner("Fetching historical stock data..."):

            try:

                result = get_historical_stock_data.invoke({
                    "ticker": ticker,
                    "period": selected_period
                })

                if isinstance(result, str):
                    result = json.loads(result)

                historical_data = result.get("historical_data", [])

                if historical_data:

                    df = pd.DataFrame(historical_data)

                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")

                    st.divider()

                    st.subheader(
                        f"{selected_company} Stock Price"
                    )

                    latest_close = df["close"].iloc[-1]
                    first_close = df["close"].iloc[0]

                    price_change = latest_close - first_close

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Latest Closing Price",
                        f"{result.get('currency', 'USD')} {latest_close:,.2f}"
                    )

                    col2.metric(
                        "Period Change",
                        f"{price_change:,.2f}",
                        f"{(price_change / first_close) * 100:.2f}%"
                        if first_close else None
                    )

                    col3.metric(
                        "Trading Days",
                        len(df)
                    )

                    st.line_chart(
                        df[["close"]],
                        y_label=f"Price ({result.get('currency', 'USD')})",
                        x_label="Date"
                    )

                    with st.expander("View Historical Data"):

                        st.dataframe(
                            df,
                            use_container_width=True
                        )

                else:

                    st.warning("No historical stock data available.")

            except Exception as e:

                st.error(f"Unable to generate chart: {e}")
