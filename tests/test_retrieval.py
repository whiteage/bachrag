import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import embedder


class EncodedVector:
    def astype(self, _dtype):
        return self


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.faiss = SimpleNamespace(normalize_L2=Mock())
        self.modules = patch.dict(sys.modules, {"faiss": self.faiss})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.model = Mock()
        self.model.encode.return_value = EncodedVector()

    def test_invalid_and_duplicate_faiss_rows_are_skipped(self):
        index = Mock(ntotal=2)
        index.search.return_value = ([[0.9, 0.8, 0.7, 0.6, 0.5]], [[-1, 1, 4, 1, 0]])
        chunks = [{"text": "first"}, {"text": "second"}]
        results = embedder.search_simple("query", index, chunks, self.model, k=20)
        self.assertEqual([item["text"] for item in results], ["second", "first"])
        self.assertEqual(index.search.call_args.args[1], 2)
        self.assertNotIn("embedding_score", chunks[0])

    def test_reranker_receives_full_text_and_scores_sort_candidates(self):
        index = Mock(ntotal=2)
        index.search.return_value = ([[0.9, 0.8]], [[0, 1]])
        chunks = [{"id": 0, "text": "a" * 1000}, {"id": 1, "text": "better"}]
        reranker = Mock()
        reranker.predict.return_value = [0.1, 0.9]
        results = embedder.search_with_rerank("query", index, chunks, self.model, reranker)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(len(reranker.predict.call_args.args[0][0][1]), 1000)
        self.assertNotIn("rerank_score", chunks[0])

    def test_zero_results_do_not_encode_or_rerank(self):
        index, reranker = Mock(ntotal=0), Mock()
        self.assertEqual(embedder.enhanced_search_with_rerank(
            "query", index, [], self.model, reranker,
        ), [])
        self.model.encode.assert_not_called()
        reranker.predict.assert_not_called()

    def test_json_storage_round_trip_and_index_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "brand.index"
            data_path = Path(directory) / "chunks.json"
            index = Mock(ntotal=1)
            self.faiss.write_index = lambda _, path: Path(path).write_bytes(b"fake-faiss")
            self.faiss.read_index = Mock(return_value=index)
            with patch.multiple(embedder, INDEX_PATH=str(index_path), DATA_PATH=str(data_path)):
                embedder.save_all(index, [{"page_content": "Текст", "metadata": {"page": 2}}])
                loaded_index, chunks = embedder.load_index_data()
                self.assertIs(loaded_index, index)
                self.assertEqual(chunks[0]["text"], "Текст")
                self.assertEqual(chunks[0]["id"], 0)
                self.assertEqual(json.loads(data_path.read_text(encoding="utf-8")), chunks)
                index.ntotal = 2
                with self.assertRaises(ValueError):
                    embedder.load_index_data()

    def test_bad_chunks_are_rejected_before_models_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('[{"text": 42}]', encoding="utf-8")
            with self.assertRaises(ValueError):
                embedder.load_chunks_from_json(path)


if __name__ == "__main__":
    unittest.main()
