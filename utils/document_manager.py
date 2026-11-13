# from pathlib import Path
# from datetime import datetime

# from config.settings import PROJECT_ROOT


# UPLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "uploaded_pdfs"


# def list_uploaded_documents(ticker):
#     company_dir = UPLOAD_DIR / ticker

#     if not company_dir.exists():
#         return []

#     documents = []

#     for file_path in company_dir.glob("*.pdf"):
#         stat = file_path.stat()

#         documents.append({
#             "filename": file_path.name,
#             "size_kb": round(stat.st_size / 1024, 2),
#             "uploaded_at": datetime.fromtimestamp(
#                 stat.st_mtime
#             ).strftime("%d-%m-%Y %H:%M"),
#             "path": str(file_path)
#         })

#     return sorted(
#         documents,
#         key=lambda doc: doc["uploaded_at"],
#         reverse=True
#     )


# This code is so that we only use supabase storage instead of chromadb.
from utils.supabase_storage import list_pdfs


def list_uploaded_documents(ticker):
    return list_pdfs(ticker)
