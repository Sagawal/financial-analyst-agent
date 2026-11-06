import streamlit as st
from rag.qa_chain import ask_question

# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="Financial Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Analyst Agent")
st.caption("Ask questions about company financial filings using RAG.")

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

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ---------------- Session State ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

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

# ---------------- User Input ----------------

question = st.chat_input("Ask a financial question...")

if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Analyzing financial documents..."):

            try:
                result = ask_question(
                    question=question,
                    ticker=ticker,
                    k=5
                )

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                # Display sources
                if sources:
                    with st.expander("View Retrieved Sources"):
                        for i, doc in enumerate(sources, start=1):
                            st.markdown(f"**Source {i}**")
                            st.write(
                                f"Ticker: {doc.metadata.get('ticker', 'N/A')}"
                            )
                            st.write(
                                f"File: {doc.metadata.get('filename', 'N/A')}"
                            )
                            st.write(
                                f"Page: {doc.metadata.get('page', 'N/A')}"
                            )
                            st.text(doc.page_content[:1000])
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")