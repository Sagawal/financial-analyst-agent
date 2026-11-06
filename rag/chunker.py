"""Section-aware chunking for SEC filings and uploaded PDF documents."""

import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# SEC filing item headings, e.g. ITEM 1A. RISK FACTORS or Item 7 - MD&A.
_ITEM_HEADING = re.compile(
    r"(?im)^\s*(ITEM\s+\d{1,2}[A-Z]?)\s*[.\-:]?\s*([^\n]{0,140})\s*$"
)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split SEC filing text at recognizable ITEM headings.

    If no reliable headings are found, return the whole text as one section.
    This is deliberately conservative: false section boundaries are worse
    than retaining a document as one generic section.
    """
    matches = list(_ITEM_HEADING.finditer(text))
    if not matches:
        return [("Document", text)] if text.strip() else []

    sections: list[tuple[str, str]] = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.append(("Preamble", preamble))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        item = match.group(1).upper()
        heading_text = match.group(2).strip(" .:-\t")
        section_name = f"{item} - {heading_text}" if heading_text else item
        content = text[start:end].strip()
        if content:
            sections.append((section_name[:180], content))

    return sections


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Chunk documents while preserving source metadata and section labels.

    SEC HTML documents are split by ITEM headings before character splitting.
    PDF page documents are split page-by-page, retaining their page metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks: list[Document] = []
    for document_index, document in enumerate(documents):
        source_type = document.metadata.get("document_type", "unknown")
        if source_type == "sec_filing":
            sections = split_into_sections(document.page_content)
        else:
            section = document.metadata.get("section", "Document")
            sections = [(section, document.page_content)]

        for section_name, section_text in sections:
            if not section_text.strip():
                continue
            section_doc = Document(
                page_content=section_text,
                metadata={**document.metadata, "section": section_name},
            )
            section_chunks = splitter.split_documents([section_doc])

            for local_index, chunk in enumerate(section_chunks):
                chunk.metadata.update({
                    "document_index": document_index,
                    "chunk_index": len(chunks),
                    "section_chunk_index": local_index,
                })
                chunks.append(chunk)

    return chunks
