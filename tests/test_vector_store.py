
import unittest

from langchain_core.documents import Document

from rag.vector_store import (
    generate_chunk_id,
    sanitize_metadata
)


class TestVectorStoreHelpers(unittest.TestCase):

    def test_same_chunk_generates_same_id(self):

        doc = Document(
            page_content="Apple revenue was 391 billion.",
            metadata={
                "ticker": "AAPL",
                "page": 10,
                "chunk_index": 1
            }
        )

        id1 = generate_chunk_id(doc)
        id2 = generate_chunk_id(doc)

        self.assertEqual(id1, id2)

    def test_different_chunks_generate_different_ids(self):

        doc1 = Document(
            page_content="Apple revenue was 391 billion.",
            metadata={"ticker": "AAPL"}
        )

        doc2 = Document(
            page_content="Apple revenue was 400 billion.",
            metadata={"ticker": "AAPL"}
        )

        self.assertNotEqual(
            generate_chunk_id(doc1),
            generate_chunk_id(doc2)
        )

    def test_metadata_sanitization(self):

        metadata = {
            "ticker": "AAPL",
            "page": 10,
            "tags": ["revenue", "profit"],
            "unused": None
        }

        result = sanitize_metadata(metadata)

        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["page"], 10)
        self.assertEqual(result["tags"], "['revenue', 'profit']")
        self.assertNotIn("unused", result)


if __name__ == "__main__":
    unittest.main()
