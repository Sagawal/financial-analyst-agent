from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# ---------------- API Configuration ----------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY", ""
).strip()

SUPABASE_STORAGE_BUCKET = "financial-reports"

LANGGRAPH_DATABASE_URL = os.getenv(
    "LANGGRAPH_DATABASE_URL", ""
).strip()

SEC_STORAGE_BUCKET = "sec-filings"


# ---------------- SEC Configuration ----------------

SEC_DATA_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

SUPPORTED_COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
}

DEFAULT_FORMS = ("10-K", "10-Q")


# ---------------- Model Configuration ----------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)


# ---------------- Directory Configuration ----------------

RAW_SEC_DIR = PROJECT_ROOT / "data" / "raw" / "sec_filings"
UPLOADED_PDF_DIR = PROJECT_ROOT / "data" / "raw" / "uploaded_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

CONVERSATION_DB_PATH = PROJECT_ROOT / "conversation_memory.sqlite"


# ---------------- Validation ----------------

def validate_sec_user_agent() -> None:
    if not SEC_USER_AGENT or "@" not in SEC_USER_AGENT:
        raise RuntimeError(
            "Set SEC_USER_AGENT in .env to an identifiable app name "
            "and contact email."
        )
