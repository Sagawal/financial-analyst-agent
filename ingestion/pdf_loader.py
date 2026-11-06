"""Load PDF files page by page using PyMuPDF."""
from pathlib import Path
import fitz
from langchain_core.documents import Document


def load_pdf(file_path: str | Path, metadata: dict | None = None) -> list[Document]:
    """Return one Document per non-empty page; page numbers are 1-based."""
    path = Path(file_path)
    base_metadata = {
        "source": str(path.resolve()), "filename": path.name,
        "document_type": "uploaded_pdf", "format": "pdf",
    }
    if metadata:
        base_metadata.update(metadata)
    documents = []
    with fitz.open(path) as pdf:
        total_pages = len(pdf)
        for page_index, page in enumerate(pdf):
            page_text = page.get_text("text").strip()
            if page_text:
                documents.append(Document(
                    page_content=page_text,
                    metadata={**base_metadata, "page": page_index + 1, "total_pages": total_pages},
                ))
    return documents
