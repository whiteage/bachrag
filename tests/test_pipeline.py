"""Exercise the web pipeline with real FAISS and deterministic model substitutes."""

import io
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import app as web
import embedder
from cfg import LLMConfig
from rag import SimpleLLMEnhancedRAG

try:
    import faiss
except ImportError:
    faiss = None


@unittest.skipIf(faiss is None, "Install faiss-cpu for the native index integration test")
class PipelineTests(unittest.TestCase):
    def test_upload_index_reload_chat_replace_and_delete(self):
        class Model:
            def encode(self, texts, **_options):
                return np.array([
                    [text.lower().count("capital"), text.lower().count("profit")]
                    for text in texts
                ], dtype="float32")

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            paths = {
                "UPLOAD_DIR": root / "uploads", "INDEX_PATH": root / "brand.index",
                "DATA_PATH": root / "chunks.json", "STATE_FILE": root / "state.json",
            }
            for name, path in paths.items():
                stack.enter_context(patch.object(web, name, path))
            stack.enter_context(patch.object(embedder, "INDEX_PATH", str(paths["INDEX_PATH"])))
            stack.enter_context(patch.object(embedder, "DATA_PATH", str(paths["DATA_PATH"])))
            stack.enter_context(patch.object(web, "_initialized", False))
            stack.enter_context(patch.object(web, "rag_system", None))
            model = Model()
            stack.enter_context(patch.object(embedder, "get_model", return_value=model))
            web.indexed_files.clear()
            web.chat_history.clear()
            web.indexing_progress.update(active=False, current="", done=0, total=0, log=[])
            client = web.app.test_client()

            def upload(text, filename):
                response = client.post(
                    "/api/upload", data={"files": (io.BytesIO(text.encode()), filename)},
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 200)

            upload("Capital 2025 is 200.", "capital.txt")
            upload("Profit 2025 is 50.", "profit.txt")
            web._run_indexing(list(web.indexed_files))
            index, chunks = embedder.load_index_data()
            self.assertEqual(index.ntotal, 2)
            self.assertEqual(embedder.search_simple("capital", index, chunks, model)[0]["source_file"], "capital.txt")

            llm = Mock()
            llm.complete.side_effect = [
                json.dumps({"subqueries": [{"subquery": "capital 2025"}]}),
                "Capital is 200 according to capital.txt.",
            ]
            reranker = Mock()
            reranker.predict.side_effect = lambda pairs, **_options: [
                0.9 if "Capital" in text else 0.2 for _query, text in pairs
            ]
            web.rag_system = SimpleLLMEnhancedRAG(
                index, chunks, model, reranker, LLMConfig(), llm_client=llm,
            )
            response = client.post("/api/chat", json={"query": "capital 2025"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("200", response.json["content"])
            self.assertIn("capital.txt", response.json["sources"])
            self.assertIn("Capital 2025 is 200.", llm.complete.call_args.args[0])

            upload("Capital 2025 is 300.", "capital.txt")
            self.assertFalse(paths["INDEX_PATH"].exists())
            web._run_indexing(list(web.indexed_files))
            index, chunks = embedder.load_index_data()
            self.assertEqual(index.ntotal, 2)
            self.assertEqual(sum("300" in item["text"] for item in chunks), 1)
            self.assertFalse(any("200" in item["text"] for item in chunks))

            self.assertEqual(client.delete("/api/files/profit.txt").status_code, 200)
            web._run_indexing(list(web.indexed_files))
            index, chunks = embedder.load_index_data()
            self.assertEqual(index.ntotal, 1)
            self.assertEqual(chunks[0]["source_file"], "capital.txt")


if __name__ == "__main__":
    unittest.main()
