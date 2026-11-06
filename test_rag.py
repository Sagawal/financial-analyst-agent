
from rag.qa_chain import ask_question

question = "What were Apple's major sources of revenue?"

result = ask_question(
    question=question,
    ticker="AAPL",
    k=5
)

print("\n========== ANSWER ==========\n")

print(result["answer"])

print("\n========== SOURCES ==========\n")

for i, doc in enumerate(result["sources"], start=1):

    print(f"\nSource {i}")
    print("Ticker:", doc.metadata.get("ticker"))
    print("File:", doc.metadata.get("filename"))
    print("Page:", doc.metadata.get("page"))
    print("Content:", doc.page_content[:300])
