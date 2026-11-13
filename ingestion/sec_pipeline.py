
from pathlib import Path
import json

from config.settings import RAW_SEC_DIR
from ingestion.html_loader import load_sec_html
from rag.chunker import chunk_documents
from utils.supabase_storage import upload_sec_html


def ingest_sec_filing(
    file_path,
    ticker,
    company,
    vector_store,
    filing_metadata=None
):
    file_path = Path(file_path)
    safe_filename = file_path.name

    # Read original HTML
    file_bytes = file_path.read_bytes()

    # Upload original HTML to Supabase Storage
    storage_path = upload_sec_html(
        ticker=ticker,
        filename=safe_filename,
        file_bytes=file_bytes
    )

    # Extract HTML content
    documents = load_sec_html(file_path)

    # Attach consistent metadata
    metadata = {
        "source": storage_path,
        "filename": safe_filename,
        "ticker": ticker,
        "company": company,
        "document_type": "sec_filing",
        "format": "html"
    }

    if filing_metadata:
        metadata.update(filing_metadata)

    # Ensure standard fields remain consistent
    metadata.update({
        "source": storage_path,
        "filename": safe_filename,
        "ticker": ticker,
        "company": company,
        "document_type": "sec_filing",
        "format": "html"
    })

    for document in documents:
        document.metadata.update(metadata)

    # Chunk and index
    chunks = chunk_documents(documents)
    vector_store.add_documents(chunks)

    return {
        "filename": safe_filename,
        "storage_path": storage_path,
        "chunks_created": len(chunks)
    }



def ingest_sec_filings_from_metadata(
    metadata_file,
    ticker,
    company,
    vector_store
):
    metadata_file = Path(metadata_file)

    with metadata_file.open("r", encoding="utf-8") as file:
        filings = json.load(file)

    results = []

    for filing in filings:
        # Process only successfully downloaded filings
        if filing.get("download_status") != "downloaded":
            continue

        file_path = Path(filing["local_path"])

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        filing_metadata = {
            "filing_type": filing.get("form"),
            "filing_date": filing.get("filing_date"),
            "report_date": filing.get("report_date"),
            "accession_number": filing.get("accession_number"),
            "description": filing.get("description"),
            "url": filing.get("url")
        }

        try:
            result = ingest_sec_filing(
                file_path=file_path,
                ticker=ticker,
                company=company,
                vector_store=vector_store,
                filing_metadata=filing_metadata
            )

            results.append(result)
            print(f"Successfully ingested: {file_path.name}")

        except Exception as e:
            print(f"Failed to ingest {file_path.name}: {e}")

    return results
