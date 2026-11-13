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
    ]

    for key in secret_keys:
        if key in secrets and secrets[key]:
            os.environ.setdefault(key, str(secrets[key]))

except Exception:
    # Local development can continue using the .env file.
    pass


# ---------------- Application Imports ----------------

from agent.graph import ask_agent
from rag.qa_chain import vector_store
from ingestion.upload_pipeline import ingest_pdf_bytes
from utils.document_manager import list_uploaded_documents
from tools.market_data import get_historical_stock_data

st.title("📊 Financial Analyst Agent")
st.caption("Ask questions about company financial filings using RAG.")



# ---------------- Conversation ID ----------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ---------------- Sidebar ----------------

st.sidebar.header("Settings")

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


# ---------------- PDF Upload ----------------

st.sidebar.subheader("Upload Financial Reports")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Ingest PDFs", disabled=not uploaded_files):

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

                st.sidebar.success(
                    f"{result['filename']}: "
                    f"{result['chunks_created']} chunks indexed."
                )

        st.sidebar.success("PDF ingestion completed!")

    except Exception as e:
        st.sidebar.error(f"Ingestion failed: {e}")


# ---------------- Document Management ----------------

st.sidebar.subheader("Uploaded Documents")

documents = list_uploaded_documents(ticker)

if not documents:
    st.sidebar.caption("No PDFs uploaded for this company yet.")

else:
    st.sidebar.caption(f"{len(documents)} PDF(s) available")

    with st.sidebar.expander("View Documents"):

        for document in documents:
            st.markdown(f"**{document['filename']}**")
            st.caption(
                f"Size: {document['size_kb']} KB | "
                f"Modified: {document['uploaded_at']}"
            )
            st.divider()


st.sidebar.divider()


# ---------------- Chat History ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()


# ---------------- Display Chat History ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("View Retrieved Sources"):

                for i, doc in enumerate(message["sources"], start=1):

                    st.markdown(f"**Source {i}**")
                    st.write(f"Ticker: {doc.metadata.get('ticker', 'N/A')}")
                    st.write(f"File: {doc.metadata.get('filename', 'N/A')}")
                    st.write(f"Page: {doc.metadata.get('page', 'N/A')}")
                    st.text(doc.page_content[:1000])
                    st.divider()


# ---------------- Chat Input ----------------

question = st.chat_input("Ask a financial question...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing financial documents..."):

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
                        "Some claims could not be verified against the retrieved sources."
                    )

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:

                    with st.expander("View Retrieved Sources"):

                        for i, doc in enumerate(sources, start=1):

                            st.markdown(f"**Source {i}**")
                            st.write(f"Ticker: {doc.metadata.get('ticker', 'N/A')}")
                            st.write(f"File: {doc.metadata.get('filename', 'N/A')}")
                            st.write(f"Page: {doc.metadata.get('page', 'N/A')}")
                            st.text(doc.page_content[:1000])
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")



# ---------------- Stock Price Chart ----------------

st.divider()
st.header("📈 Stock Price History")

col1, col2 = st.columns([1, 2])

with col1:
    selected_period = st.selectbox(
        "Select Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=0
    )

with col2:
    st.write(f"Historical stock prices for **{selected_company} ({ticker})**")

if st.button("Generate Stock Chart"):

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

                st.subheader(f"{selected_company} Stock Price (USD)")

                st.line_chart(
                    df[["close"]],
                    y_label="Price (USD)",
                    x_label="Date"
                )

                with st.expander("View Historical Data"):
                    st.dataframe(df, use_container_width=True)

            else:
                st.warning("No historical stock data available.")

        except Exception as e:
            st.error(f"Unable to generate chart: {e}")