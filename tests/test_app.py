import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import app as web
import embedder


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.patches = [
            patch.object(web, "UPLOAD_DIR", self.directory / "uploads"),
            patch.object(web, "INDEX_PATH", self.directory / "brand.index"),
            patch.object(web, "DATA_PATH", self.directory / "chunks.json"),
            patch.object(web, "STATE_FILE", self.directory / "state.json"),
            patch.object(web, "_initialized", False),
            patch.object(web, "rag_system", None),
            patch.dict(os.environ, {}, clear=False),
        ]
        for item in self.patches:
            item.start()
        web.indexed_files.clear()
        web.chat_history.clear()
        web.indexing_progress.update(active=False, current="", done=0, total=0, log=[])
        self.client = web.app.test_client()
        web.app.config["TESTING"] = True

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def upload(self, text="document text", filename="document.txt"):
        return self.client.post(
            "/api/upload", data={"files": (io.BytesIO(text.encode()), filename)},
            content_type="multipart/form-data",
        )

    def test_template_and_assets_are_separate(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"/static/app.css", response.data)
        self.assertIn(b"/static/app.js", response.data)
        with self.client.get("/static/app.js") as asset:
            self.assertEqual(asset.status_code, 200)

    def test_upload_validates_formats_and_preserves_cyrillic_basename(self):
        self.assertEqual(self.client.post("/api/upload").status_code, 400)
        self.assertEqual(self.upload(filename="script.exe").status_code, 400)
        response = self.upload(filename="../../отчёт.txt")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["uploaded"][0]["name"], "отчёт.txt")
        self.assertTrue((web.UPLOAD_DIR / "отчёт.txt").is_file())
        with self.assertRaises(ValueError):
            web._safe_path("../state.json")
        with self.assertRaises(ValueError):
            web._safe_path("..\\state.json")

    def test_json_validation_does_not_raise_for_wrong_types(self):
        for payload in (None, [], {}, {"query": 123}, {"query": " "}):
            with self.subTest(payload=payload):
                response = self.client.post("/api/chat", json=payload)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.post("/api/chat", json={"query": "hello"}).status_code, 503)
        for payload in ([], {}, {"api_key": 123}, {"api_key": ""}):
            self.assertEqual(self.client.post("/api/config", json=payload).status_code, 400)

    def test_api_key_stays_in_process_and_response_does_not_echo_it(self):
        secret = "test-runtime-credential"
        response = self.client.post("/api/config", json={"api_key": secret})
        self.assertEqual(response.json, {"ok": True})
        self.assertEqual(os.environ["LLM_API_KEY"], secret)
        self.assertNotIn(secret, web.STATE_FILE.read_text(encoding="utf-8"))
        self.assertNotIn(secret.encode(), self.client.get("/api/index/status").data)
        cached = object()
        web.rag_system = cached
        self.client.post("/api/config", json={"api_key": secret})
        self.assertIs(web.rag_system, cached)

    def test_active_flag_is_set_before_worker_starts(self):
        self.upload()
        with patch.object(web.threading, "Thread") as worker:
            self.assertEqual(self.client.post("/api/index").status_code, 200)
            worker.return_value.start.assert_called_once()
            self.assertEqual(self.client.post("/api/index").status_code, 409)
            self.assertEqual(self.upload(filename="other.txt").status_code, 409)
            self.assertEqual(self.client.delete("/api/files/document.txt").status_code, 409)

    def test_delete_invalidates_cached_index_and_remaining_documents(self):
        self.upload(filename="first.txt")
        self.upload(filename="second.txt")
        web.INDEX_PATH.write_text("index")
        web.DATA_PATH.write_text("[]")
        web.rag_system = object()
        for metadata in web.indexed_files.values():
            metadata["status"] = "done"
        self.assertEqual(self.client.delete("/api/files/first.txt").status_code, 200)
        self.assertFalse(web.INDEX_PATH.exists())
        self.assertFalse(web.DATA_PATH.exists())
        self.assertIsNone(web.rag_system)
        self.assertEqual(web.indexed_files["second.txt"]["status"], "pending")
        self.assertEqual(self.client.delete("/api/files/first.txt").status_code, 404)

    def test_reindex_replaces_modified_chunks_and_encodes_once(self):
        self.upload("old text")
        saved = []

        class Embeddings:
            shape = (1, 2)

            def astype(self, _dtype):
                return self

        class Index:
            def __init__(self, _dimension):
                self.ntotal = 0

            def add(self, values):
                self.ntotal = values.shape[0]

        model = Mock()
        model.encode.return_value = Embeddings()
        faiss = types.SimpleNamespace(normalize_L2=Mock(), IndexFlatIP=Index)

        def save_index(_index, chunks):
            saved.append(chunks)
            web.INDEX_PATH.write_text("index")
            web.DATA_PATH.write_text(json.dumps(chunks))

        with patch.dict(sys.modules, {"faiss": faiss}), patch.object(embedder, "get_model", return_value=model), patch.object(embedder, "save_all", side_effect=save_index):
            web._run_indexing(["document.txt"])
            self.assertEqual(web.indexed_files["document.txt"]["status"], "done")
            self.assertEqual(model.encode.call_count, 1)
            self.upload("replacement text")
            self.assertFalse(web.DATA_PATH.exists())
            web._run_indexing(["document.txt"])
        self.assertEqual(model.encode.call_count, 2)
        self.assertEqual([chunk["text"] for chunk in saved[-1]], ["replacement text"])
        self.assertEqual([chunk["id"] for chunk in saved[-1]], [0])

    def test_docx_text_and_tables_use_local_parser(self):
        self.client.get("/api/files")
        path = web.UPLOAD_DIR / "document.docx"
        with ZipFile(path, "w") as archive:
            archive.writestr(
                "word/document.xml",
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>Paragraph</w:t><w:tab/><w:t>2024</w:t>'
                '<w:br/><w:t>123</w:t><w:cr/><w:t>456</w:t></w:r></w:p>'
                '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell</w:t></w:r></w:p>'
                '</w:tc></w:tr></w:tbl></w:body></w:document>',
            )
        chunks = web._read_document(path)
        self.assertIn("Paragraph\t2024\n123\n456", chunks[0]["text"])
        self.assertIn("Cell", chunks[0]["text"])

    def test_chat_keeps_bounded_history_and_returns_sources(self):
        self.client.get("/api/files")
        web.rag_system = Mock()
        web.rag_system.process_complex_query.return_value = {
            "synthesized_answer": "Answer", "subqueries": [{"subquery": "question"}],
            "results": {"question": [{"metadata": {"source": "report.pdf", "page": 2}}]},
        }
        response = self.client.post("/api/chat", json={"query": "question"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["sources"], ["report.pdf стр.2"])
        self.assertEqual(len(self.client.get("/api/chat/history").json), 2)
        self.assertEqual(self.client.post("/api/chat/clear").json, {"ok": True})
        self.assertEqual(self.client.get("/api/chat/history").json, [])


if __name__ == "__main__":
    unittest.main()
