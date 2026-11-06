"""Download primary 10-K/10-Q filing documents from SEC EDGAR."""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

import requests

from config.settings import (
    DEFAULT_FORMS,
    RAW_SEC_DIR,
    SEC_ARCHIVES_BASE_URL,
    SEC_DATA_BASE_URL,
    SEC_USER_AGENT,
    SUPPORTED_COMPANIES,
    validate_sec_user_agent,
)

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = 0.15


def normalize_cik(cik: int | str) -> str:
    """Return a SEC CIK as a zero-padded 10-character string."""
    return str(cik).strip().zfill(10)


def accession_to_path(accession_number: str) -> str:
    """SEC Archives accession directory omits hyphens."""
    return accession_number.replace("-", "")


def build_filing_url(cik: int | str, accession_number: str, primary_document: str) -> str:
    cik_without_zeros = str(int(normalize_cik(cik)))
    accession_path = accession_to_path(accession_number)
    return (
        f"{SEC_ARCHIVES_BASE_URL}/{cik_without_zeros}/"
        f"{accession_path}/{primary_document}"
    )


def make_session() -> requests.Session:
    validate_sec_user_agent()
    session = requests.Session()
    session.headers.update({
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def get_company_ciks(session: requests.Session) -> dict[str, str]:
    """Fetch SEC's ticker mapping and return ticker -> zero-padded CIK."""
    response = session.get(TICKER_MAP_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    mapping = {}
    for item in payload.values():
        ticker = str(item.get("ticker", "")).upper()
        if ticker:
            mapping[ticker] = normalize_cik(item["cik_str"])
    return mapping


def get_recent_filings(
    session: requests.Session,
    cik: str,
    forms: Iterable[str],
    limit: int,
) -> list[dict]:
    """Return up to `limit` recent filings, across the requested form types."""
    url = f"{SEC_DATA_BASE_URL}/submissions/CIK{normalize_cik(cik)}.json"
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    recent = response.json().get("filings", {}).get("recent", {})

    requested_forms = set(forms)
    rows = []
    count = len(recent.get("form", []))

    for index in range(count):
        form = recent["form"][index]
        if form not in requested_forms:
            continue

        primary_document = recent["primaryDocument"][index]
        accession = recent["accessionNumber"][index]
        if not primary_document or not accession:
            continue

        rows.append({
            "form": form,
            "filing_date": recent["filingDate"][index],
            "report_date": recent.get("reportDate", [""] * count)[index],
            "accession_number": accession,
            "primary_document": primary_document,
            "description": recent.get("primaryDocDescription", [""] * count)[index],
            "url": build_filing_url(cik, accession, primary_document),
        })
        if len(rows) >= limit:
            break

    return rows


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def download_company_filings(
    session: requests.Session,
    ticker: str,
    cik: str,
    forms: Iterable[str] = DEFAULT_FORMS,
    limit: int = 2,
    output_dir: Path = RAW_SEC_DIR,
) -> list[dict]:
    company_dir = output_dir / ticker
    company_dir.mkdir(parents=True, exist_ok=True)

    filings = get_recent_filings(session, cik, forms, limit)
    downloaded = []

    for filing in filings:
        filename = safe_filename(
            f"{filing['filing_date']}_{filing['form']}_{filing['accession_number']}_"
            f"{Path(filing['primary_document']).name}"
        )
        destination = company_dir / filename

        if destination.exists() and destination.stat().st_size > 0:
            filing["local_path"] = str(destination)
            filing["download_status"] = "already_downloaded"
            downloaded.append(filing)
            continue

        response = session.get(filing["url"], timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        destination.write_bytes(response.content)

        filing["local_path"] = str(destination)
        filing["download_status"] = "downloaded"
        downloaded.append(filing)
        time.sleep(REQUEST_DELAY_SECONDS)

    metadata_path = company_dir / "filings_metadata.json"
    metadata_path.write_text(
        json.dumps(downloaded, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SEC 10-K/10-Q filings.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Download for AAPL, MSFT, NVDA.")
    group.add_argument("--ticker", choices=sorted(SUPPORTED_COMPANIES), help="Company ticker.")
    parser.add_argument(
        "--forms", nargs="+", choices=["10-K", "10-Q"], default=list(DEFAULT_FORMS),
        help="Filing forms to download (default: 10-K 10-Q).",
    )
    parser.add_argument("--limit", type=int, default=2, help="Maximum filings per company.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    validate_sec_user_agent()
    session = make_session()

    print("Looking up company CIKs from SEC...")
    ticker_to_cik = get_company_ciks(session)

    tickers = sorted(SUPPORTED_COMPANIES) if args.all else [args.ticker]
    for ticker in tickers:
        if ticker not in ticker_to_cik:
            print(f"[SKIP] Could not find CIK for {ticker}")
            continue

        print(f"\n{ticker} — {SUPPORTED_COMPANIES[ticker]}")
        filings = download_company_filings(
            session=session,
            ticker=ticker,
            cik=ticker_to_cik[ticker],
            forms=args.forms,
            limit=args.limit,
        )
        for filing in filings:
            print(
                f"[{filing['download_status']}] {filing['form']} "
                f"{filing['filing_date']} -> {filing['local_path']}"
            )

    print("\nSEC download step finished.")


if __name__ == "__main__":
    main()
