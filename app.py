import streamlit as st

st.set_page_config(page_title="Financial Analyst Agent", page_icon="📊", layout="wide")
st.title("📊 Financial Analyst Agent")
st.info(
    "Phase 1 is under development. The first milestone is SEC filing ingestion; "
    "the RAG chat interface will be connected in a later step."
)
st.write("Supported companies: Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA).")
