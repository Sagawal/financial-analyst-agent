from pathlib import Path

import fitz
from langchain_core.documents import Document

from config.settings import PROJECT_ROOT
from rag.chunker import chunk_documents


UPLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "uploaded_pdfs"


def ingest_pdf_bytes(
    file_bytes,
    filename,
    ticker,
    company,
    vector_store
):
    # Save the original PDF
    company_dir = UPLOAD_DIR / ticker
    company_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(filename).name
    file_path = company_dir / safe_filename
    file_path.write_bytes(file_bytes)

    # Extract text page by page
    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    documents = []

    try:
        total_pages = len(pdf)

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "filename": safe_filename,
                        "ticker": ticker,
                        "company": company,
                        "document_type": "uploaded_pdf",
                        "format": "pdf",
                        "filing_type": "PDF Upload",
                        "page": page_number,
                        "total_pages": total_pages
                    }
                )
            )
    finally:
        pdf.close()

    if not documents:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned document requiring OCR."
        )

    # Split documents into chunks
    chunks = chunk_documents(documents)

    # Store chunks in ChromaDB
    vector_store.add_documents(chunks)

    return {
        "filename": safe_filename,
        "pages_extracted": len(documents),
        "chunks_created": len(chunks)
    }