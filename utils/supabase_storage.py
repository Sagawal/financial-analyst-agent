
from datetime import datetime
from functools import lru_cache

from supabase import create_client

from config.settings import (
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_STORAGE_BUCKET,
)


@lru_cache(maxsize=1)
def get_supabase_client():

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError(
            "SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing."
        )

    return create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_ROLE_KEY
    )


def upload_pdf(ticker: str, filename: str, file_bytes: bytes):

    client = get_supabase_client()

    storage_path = f"{ticker}/{filename}"

    client.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
        path=storage_path,
        file=file_bytes,
        file_options={
            "content-type": "application/pdf",
            "upsert": "false"
        }
    )

    return storage_path



def upload_sec_html(ticker, filename, file_bytes):
    client = get_supabase_client()

    storage_path = f"{ticker}/{filename}"

    client.storage.from_("sec-filings").upload(
        path=storage_path,
        file=file_bytes,
        file_options={
            "content-type": "text/html",
            "upsert": "true"
        }
    )

    return storage_path


def list_pdfs(ticker: str):

    client = get_supabase_client()

    files = client.storage.from_(
        SUPABASE_STORAGE_BUCKET
    ).list(path=ticker)

    documents = []

    for file in files:

        filename = file.get("name", "")

        if not filename.lower().endswith(".pdf"):
            continue

        metadata = file.get("metadata") or {}

        uploaded_at = file.get("created_at") or file.get("updated_at")

        if uploaded_at:
            try:
                date = datetime.fromisoformat(
                    uploaded_at.replace("Z", "+00:00")
                )
                uploaded_at = date.strftime("%d-%m-%Y %H:%M")
            except ValueError:
                pass

        documents.append({
            "filename": filename,
            "size_kb": round(
                int(metadata.get("size", 0)) / 1024, 2
            ),
            "uploaded_at": uploaded_at or "Unknown",
            "path": f"{ticker}/{filename}"
        })

    return sorted(
        documents,
        key=lambda doc: doc["uploaded_at"],
        reverse=True
    )
