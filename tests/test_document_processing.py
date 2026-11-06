import unittest
from langchain_core.documents import Document
from rag.document_processor import clean_text, process_documents


class TestDocumentProcessor(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(
            clean_text("  Revenue\u00a0\t growth  \n\n\n strong "),
            "Revenue growth\n\nstrong"
        )

    def test_process_documents_preserves_metadata(self):
        original = Document(page_content="  Apple's revenue was $100 billion.  ",
                            metadata={"ticker": "AAPL", "page": 12, "document_type": "uploaded_pdf"})
        result = process_documents([original])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].page_content, "Apple's revenue was $100 billion.")
        self.assertEqual(result[0].metadata["ticker"], "AAPL")
        self.assertEqual(result[0].metadata["page"], 12)
        self.assertEqual(result[0].metadata["company"], "unknown")


if __name__ == "__main__":
    unittest.main()
