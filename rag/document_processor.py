"""Common text cleaning and metadata normalization for ingested documents."""
import re
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Normalize whitespace and common PDF/HTML extraction artifacts."""

    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")

    # Clean each line individually
    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line)
        lines.append(line.strip())

    text = "\n".join(lines)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def process_documents(documents: list[Document]) -> list[Document]:
    """Clean text while preserving source-specific metadata."""
    processed = []
    for document in documents:
        content = clean_text(document.page_content)
        if not content:
            continue
        metadata = {
            "company": document.metadata.get("company", "unknown"),
            "ticker": document.metadata.get("ticker", "unknown"),
            "document_type": document.metadata.get("document_type", "unknown"),
            **document.metadata,
        }
        processed.append(Document(page_content=content, metadata=metadata))
    return processed
