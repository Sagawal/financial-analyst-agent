import unittest
from langchain_core.documents import Document
from rag.chunker import chunk_documents, split_into_sections


class TestChunker(unittest.TestCase):
    def test_sec_item_sections_are_detected(self):
        text = (
            "Company introduction\n"
            "ITEM 1. BUSINESS\nApple makes products.\n"
            "ITEM 1A. RISK FACTORS\nCompetition is a risk.\n"
            "ITEM 7. MANAGEMENT DISCUSSION AND ANALYSIS\nRevenue increased."
        )
        sections = split_into_sections(text)
        self.assertEqual(len(sections), 4)
        self.assertTrue(sections[1][0].startswith("ITEM 1 -"))
        self.assertTrue(sections[2][0].startswith("ITEM 1A -"))

    def test_chunk_size_and_metadata(self):
        doc = Document(
            page_content="Revenue increased. " * 100,
            metadata={"ticker": "AAPL", "document_type": "uploaded_pdf", "page": 4},
        )
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.page_content) <= 100 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["ticker"] == "AAPL" for chunk in chunks))
        self.assertTrue(all(chunk.metadata["page"] == 4 for chunk in chunks))

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_documents([], chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    unittest.main()
