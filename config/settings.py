from pathlib import Path
import os
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "").strip()
SEC_DATA_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

SUPPORTED_COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
}

DEFAULT_FORMS = ("10-K", "10-Q")
RAW_SEC_DIR = PROJECT_ROOT / "data" / "raw" / "sec_filings"


def validate_sec_user_agent() -> None:
    if not SEC_USER_AGENT or "@" not in SEC_USER_AGENT:
        raise RuntimeError(
            "Set SEC_USER_AGENT in .env to an identifiable app name and valid "
            "contact email, e.g. FinancialAnalystAgent/0.1 name@example.com"
        )
