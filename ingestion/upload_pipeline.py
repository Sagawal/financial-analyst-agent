from pathlib import Path

import fitz
from langchain_core.documents import Document

from config.settings import UPLOADED_PDF_DIR
from rag.chunker import chunk_documents
from utils.supabase_storage import upload_pdf


def ingest_pdf_bytes(
    file_bytes,
    filename,
    ticker,
    company,
    vector_store
):

    safe_filename = Path(filename).name

    # Extract PDF text page by page
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
                        "source": f"{ticker}/{safe_filename}",
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

    # Upload original PDF to Supabase Storage
    storage_path = upload_pdf(
        ticker=ticker,
        filename=safe_filename,
        file_bytes=file_bytes
    )

    # Keep a local copy for local development
    company_dir = UPLOADED_PDF_DIR / ticker
    company_dir.mkdir(parents=True, exist_ok=True)

    file_path = company_dir / safe_filename
    file_path.write_bytes(file_bytes)

    # Chunk extracted documents
    chunks = chunk_documents(documents)

    # Store chunks and embeddings in Supabase vector store
    vector_store.add_documents(chunks)

    return {
        "filename": safe_filename,
        "storage_path": storage_path,
        "pages_extracted": len(documents),
        "chunks_created": len(chunks)
    }