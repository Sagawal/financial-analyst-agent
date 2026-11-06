"""Load downloaded SEC filing HTML into LangChain Documents."""
from pathlib import Path
from typing import Any
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def _table_to_text(table: Any) -> str:
    """Convert an HTML table to readable pipe-separated rows."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [" ".join(cell.stripped_strings).replace("|", r"\|")
                 for cell in tr.find_all(["th", "td"], recursive=False)]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def load_sec_html(file_path: str | Path, metadata: dict | None = None) -> list[Document]:
    """Extract filing text while retaining basic table content and metadata."""
    path = Path(file_path)
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for table in soup.find_all("table"):
        table.replace_with(soup.new_string("\n[TABLE]\n" + _table_to_text(table) + "\n[/TABLE]\n"))
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        heading.insert_before("\n")
        heading.insert_after("\n")
    text = soup.get_text(separator=" ", strip=True)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    doc_metadata = {
        "source": str(path.resolve()), "filename": path.name,
        "document_type": "sec_filing", "format": "html",
    }
    if metadata:
        doc_metadata.update(metadata)
    return [Document(page_content=text, metadata=doc_metadata)]
