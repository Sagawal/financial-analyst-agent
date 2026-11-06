import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from rag.embeddings import BGEEmbeddings


class TestBGEEmbeddings(unittest.TestCase):
    @patch("rag.embeddings.SentenceTransformer")
    def test_embed_documents(self, sentence_transformer):
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        sentence_transformer.return_value = model
        embeddings = BGEEmbeddings()
        result = embeddings.embed_documents(["chunk one", "chunk two"])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 3)
        self.assertTrue(model.encode.call_args.kwargs["normalize_embeddings"])

    @patch("rag.embeddings.SentenceTransformer")
    def test_embed_query_adds_instruction(self, sentence_transformer):
        model = MagicMock()
        model.encode.return_value = np.array([0.1, 0.2, 0.3])
        sentence_transformer.return_value = model
        embeddings = BGEEmbeddings()
        result = embeddings.embed_query("What was Apple's revenue?")
        self.assertEqual(len(result), 3)
        query = model.encode.call_args.args[0]
        self.assertTrue(query.startswith(BGEEmbeddings.QUERY_INSTRUCTION))
        self.assertTrue(query.endswith("What was Apple's revenue?"))

    @patch("rag.embeddings.SentenceTransformer")
    def test_empty_documents(self, sentence_transformer):
        embeddings = BGEEmbeddings()
        self.assertEqual(embeddings.embed_documents([]), [])


if __name__ == "__main__":
    unittest.main()
