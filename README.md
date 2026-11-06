# Financial Analyst Agent

A financial-document research assistant built incrementally with Python, SEC EDGAR, Groq, HuggingFace BGE, ChromaDB, LangGraph, and Streamlit.

## Initial scope

The first implementation step provides:
- Project configuration
- SEC company lookup for AAPL, MSFT, and NVDA
- Download of recent 10-K and 10-Q primary filing documents
- Filing metadata saved alongside each downloaded document
- Basic unit tests for URL and CIK handling

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

**Windows PowerShell**
```powershell
Copy-Item .env.example .env
```

**macOS/Linux**
```bash
cp .env.example .env
```

Edit `.env` and set `SEC_USER_AGENT` to an identifiable application name and a valid contact email, for example:

```text
SEC_USER_AGENT=FinancialAnalystAgent/0.1 sagar@example.com
```

The SEC requires a descriptive User-Agent for automated access. Do not use a fake or blank contact address.

## Download SEC filings

Download the latest 2 10-K/10-Q filings for all configured companies:

```bash
python -m ingestion.sec_downloader --all --limit 2
```

Download filings for one company:

```bash
python -m ingestion.sec_downloader --ticker AAPL --limit 4
```

Download only annual reports:

```bash
python -m ingestion.sec_downloader --ticker MSFT --forms 10-K --limit 3
```

Files are saved under:

```text
data/raw/sec_filings/<TICKER>/
```

A `filings_metadata.json` file in each company folder records the filing details and local filenames.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Important notes

- SEC submissions data is used to discover filing accessions and primary document names.
- This first version downloads the primary filing document (usually HTML), not every exhibit attached to a filing.
- The SEC may return HTTP 403/429 if requests are too frequent or the User-Agent is missing/invalid. Run downloads responsibly and avoid parallel bulk requests.
- Financial figures and filing periods will need careful normalization in later RAG steps.
