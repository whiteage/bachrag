"""Local Flask interface for uploading, indexing and querying documents."""

import json
import os
import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from cfg import LLMConfig, settings

UPLOAD_DIR = settings.upload_dir
INDEX_PATH = settings.index_path
DATA_PATH = settings.data_path
STATE_FILE = settings.state_file
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# One process owns the local index. Model imports and downloads happen only when
# indexing or querying, so importing the WSGI application stays inexpensive.
state_lock = threading.RLock()
rag_lock = threading.RLock()
rag_system = None
indexed_files: dict[str, dict] = {}
indexing_progress = {"active": False, "current": "", "done": 0, "total": 0, "log": []}
chat_history: list[dict] = []
_initialized = False


def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def fmt_size(size: int) -> str:
    for unit in ("Б", "КБ", "МБ"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} ГБ"


def log_progress(message: str) -> None:
    with state_lock:
        indexing_progress["log"].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        indexing_progress["log"] = indexing_progress["log"][-200:]


def _safe_path(filename: str) -> Path:
    """Accept a stored basename and reject traversal and symlink escapes."""
    if not filename or filename != Path(filename).name or "\\" in filename:
        raise ValueError("Некорректное имя файла")
    path = UPLOAD_DIR / filename
    if path.resolve().parent != UPLOAD_DIR.resolve() or path.is_symlink():
        raise ValueError("Некорректное имя файла")
    return path


def _upload_name(filename: str) -> str:
    # Keep Cyrillic document names while removing path components and characters
    # that Windows cannot store safely.
    basename = unicodedata.normalize("NFKC", filename).replace("\\", "/").rsplit("/", 1)[-1]
    basename = re.sub(r"[^\w.() -]", "_", basename).strip(" .")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if basename.partition(".")[0].upper() in reserved:
        basename = "_" + basename
    return basename


def save_files_state() -> None:
    with state_lock:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"files": indexed_files}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(STATE_FILE)


def _index_exists() -> bool:
    return INDEX_PATH.is_file() and DATA_PATH.is_file()


def _invalidate_index() -> None:
    """An edited corpus must never answer from obsolete document chunks."""
    global rag_system
    with rag_lock:
        rag_system = None
        INDEX_PATH.unlink(missing_ok=True)
        DATA_PATH.unlink(missing_ok=True)
    for metadata in indexed_files.values():
        metadata.update(status="pending", indexed_at=None)


def _sync_files() -> None:
    changed = False
    for name in list(indexed_files):
        try:
            path = _safe_path(name)
        except ValueError:
            indexed_files.pop(name)
            changed = True
            continue
        if not path.is_file():
            indexed_files.pop(name)
            changed = True
    for path in UPLOAD_DIR.iterdir():
        if not path.is_file() or path.is_symlink() or not allowed(path.name):
            continue
        stat = path.stat()
        metadata = indexed_files.get(path.name)
        if metadata is None or metadata.get("modified_ns") != stat.st_mtime_ns:
            indexed_files[path.name] = {
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "status": "pending",
                "indexed_at": None,
            }
            changed = True
    if changed:
        _invalidate_index()


def load_files_state() -> None:
    if STATE_FILE.is_file():
        try:
            files = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("files", {})
            if isinstance(files, dict):
                indexed_files.update({name: meta for name, meta in files.items() if isinstance(meta, dict)})
        except (OSError, ValueError, AttributeError):
            log_progress("⚠️ Не удалось прочитать состояние документов; индекс будет пересоздан.")
            _invalidate_index()
    _sync_files()
    if not _index_exists():
        _invalidate_index()
    for metadata in indexed_files.values():
        if metadata.get("status") == "indexing":
            metadata["status"] = "pending"


@app.before_request
def initialize() -> None:
    global _initialized
    with state_lock:
        if _initialized:
            return
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        load_files_state()
        save_files_state()
        _initialized = True


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return jsonify(error="Размер загрузки превышает 200 МБ"), 413


def _make_rag(index, data, model, reranker):
    from rag import SimpleLLMEnhancedRAG

    return SimpleLLMEnhancedRAG(
        index=index,
        data=data,
        embedding_model=model,
        reranker=reranker,
        llm_config=LLMConfig.from_env(),
    )


def try_load_rag() -> bool:
    global rag_system
    if rag_system is not None:
        return True
    if not _index_exists():
        return False
    if not LLMConfig.from_env().api_key:
        return False
    try:
        from embedder import get_model, get_reranker, load_index_data

        index, data = load_index_data()
        rag_system = _make_rag(index, data, get_model(), get_reranker())
        return True
    except Exception:
        app.logger.exception("Could not initialize the RAG system")
        return False


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="Файлы не переданы"), 400
    uploaded = []
    with state_lock:
        if indexing_progress["active"]:
            return jsonify(error="Дождитесь завершения индексирования"), 409
        for file in files:
            name = _upload_name(file.filename or "")
            if not name or not allowed(name):
                continue
            try:
                destination = _safe_path(name)
            except ValueError:
                continue
            temporary = destination.with_name(destination.name + ".upload")
            file.save(temporary)
            temporary.replace(destination)
            stat = destination.stat()
            indexed_files[name] = {
                "size": stat.st_size, "modified_ns": stat.st_mtime_ns,
                "status": "pending", "indexed_at": None,
            }
            uploaded.append({"name": name, "size": fmt_size(stat.st_size), "status": "pending"})
        if not uploaded:
            return jsonify(error="Нет файлов поддерживаемого формата"), 400
        _invalidate_index()
        save_files_state()
    return jsonify(uploaded=uploaded)


@app.get("/api/files")
def list_files():
    with state_lock:
        if not indexing_progress["active"]:
            _sync_files()
            save_files_state()
        result = [
            {"name": name, "size": fmt_size(meta["size"]),
             "status": meta["status"], "indexed_at": meta.get("indexed_at")}
            for name, meta in indexed_files.items()
        ]
    order = {"pending": 0, "modified": 1, "indexing": 2, "error": 3, "done": 4}
    result.sort(key=lambda item: (order.get(item["status"], 9), item["name"]))
    return jsonify(result)


@app.delete("/api/files/<filename>")
def delete_file(filename: str):
    try:
        path = _safe_path(filename)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    with state_lock:
        if indexing_progress["active"]:
            return jsonify(error="Дождитесь завершения индексирования"), 409
        if not path.is_file():
            return jsonify(error="Файл не найден"), 404
        path.unlink()
        indexed_files.pop(filename, None)
        _invalidate_index()
        save_files_state()
    return jsonify(ok=True)


def _text_chunks(text: str, source: str) -> list[dict]:
    return [
        {"text": text[offset:offset + 2000].strip(), "metadata": {"source": source}}
        for offset in range(0, len(text), 1800)
        if text[offset:offset + 2000].strip()
    ]


def _read_document(path: Path) -> list[dict]:
    if path.suffix.lower() == ".pdf":
        from document_processing import process_pdf

        documents = process_pdf(str(path))
        chunks = []
        for document in documents:
            if isinstance(document, dict):
                text = document.get("text") or document.get("page_content", "")
                metadata = document.get("metadata", {})
            else:
                text, metadata = document.page_content, document.metadata
            if isinstance(text, str) and text.strip():
                chunks.append({"text": text.strip(), "metadata": {**metadata, "source": path.name}})
        return chunks
    if path.suffix.lower() == ".docx":
        with ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        word_namespace = "{" + namespace["w"] + "}"
        separators = {word_namespace + "tab": "\t", word_namespace + "br": "\n", word_namespace + "cr": "\n"}
        text = "\n".join(
            "".join(
                element.text or "" if element.tag == word_namespace + "t"
                else separators.get(element.tag, "")
                for element in paragraph.iter()
            )
            for paragraph in root.findall(".//w:p", namespace)
        )
    else:
        text = path.read_text(encoding="utf-8-sig")
    return _text_chunks(text, path.name)


def _run_indexing(files: list[str]) -> None:
    global rag_system
    completed = []
    try:
        chunks = []
        for filename in files:
            with state_lock:
                indexing_progress["current"] = filename
                indexed_files[filename]["status"] = "indexing"
            try:
                file_chunks = _read_document(_safe_path(filename))
                if not file_chunks:
                    raise ValueError("Документ не содержит текста")
                for chunk in file_chunks:
                    chunk.update(id=len(chunks), source_file=filename)
                    chunks.append(chunk)
                completed.append(filename)
                log_progress(f"✅ {filename}: {len(file_chunks)} чанков")
            except Exception:
                app.logger.exception("Document processing failed: %s", filename)
                with state_lock:
                    indexed_files[filename]["status"] = "error"
                log_progress(f"❌ Не удалось обработать {filename}")
            with state_lock:
                indexing_progress["done"] += 1
        if not chunks:
            return
        import faiss

        from embedder import get_model, save_all

        model = get_model()
        log_progress(f"🔢 Создание эмбеддингов: {len(chunks)} чанков")
        embeddings = model.encode(
            [chunk["text"] for chunk in chunks], batch_size=8,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        save_all(index, chunks)
        # Indexing works without an API key. The language client is constructed
        # when the user first asks a question, after configuration is available.
        with state_lock:
            with rag_lock:
                rag_system = None
            for filename in completed:
                indexed_files[filename].update(status="done", indexed_at=time.strftime("%H:%M %d.%m"))
        log_progress("🚀 Индекс сохранён и готов к поиску")
    except Exception:
        app.logger.exception("Indexing failed")
        log_progress("❌ Индексирование не завершено. Проверьте зависимости и повторите попытку.")
        with state_lock:
            for filename in completed:
                indexed_files[filename]["status"] = "error"
    finally:
        with state_lock:
            indexing_progress.update(active=False, current="")
            save_files_state()


@app.post("/api/index")
def start_indexing():
    with state_lock:
        if indexing_progress["active"]:
            return jsonify(error="Индексирование уже запущено"), 409
        _sync_files()
        if not any(meta["status"] in {"pending", "modified", "error"} for meta in indexed_files.values()):
            return jsonify(error="Нет файлов для индексирования"), 400
        files = list(indexed_files)
        _invalidate_index()
        indexing_progress.update(active=True, current="", done=0, total=len(files), log=[])
        try:
            threading.Thread(target=_run_indexing, args=(files,), daemon=True).start()
        except RuntimeError:
            indexing_progress["active"] = False
            return jsonify(error="Не удалось запустить индексирование"), 500
    return jsonify(started=True, files=files)


@app.get("/api/index/status")
def index_status():
    with state_lock:
        return jsonify(**indexing_progress, rag_ready=rag_system is not None or _index_exists())


def _json_object():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


@app.post("/api/chat")
def chat():
    data = _json_object()
    query = data.get("query") if data is not None else None
    if not isinstance(query, str) or not query.strip():
        return jsonify(error="Введите непустой текстовый запрос"), 400
    query = query.strip()
    if len(query) > 10000:
        return jsonify(error="Запрос слишком длинный"), 400
    with rag_lock:
        if not try_load_rag():
            return jsonify(error="RAG не готов. Проверьте API ключ и проиндексируйте документы."), 503
        try:
            result = rag_system.process_complex_query(query)
        except Exception:
            app.logger.exception("Query failed")
            return jsonify(error="Не удалось получить ответ. Проверьте настройки API."), 502
    sources = []
    for matches in result.get("results", {}).values():
        for match in matches[:2]:
            metadata = match.get("metadata", {})
            source = metadata.get("source") or match.get("source_file") or metadata.get("table") or "документ"
            label = Path(str(source)).name
            if metadata.get("page") is not None:
                label += f" стр.{metadata['page']}"
            if label not in sources:
                sources.append(label)
    timestamp = time.strftime("%H:%M")
    message = {
        "id": uuid.uuid4().hex[:8], "role": "assistant",
        "content": result.get("synthesized_answer", "Ответ не найден."),
        "sources": sources[:6],
        "subqueries": [item.get("subquery", "") for item in result.get("subqueries", [])],
        "ts": timestamp,
    }
    with state_lock:
        chat_history.extend([
            {"id": uuid.uuid4().hex[:8], "role": "user", "content": query, "ts": timestamp}, message,
        ])
        del chat_history[:-100]
    return jsonify(message)


@app.get("/api/chat/history")
def get_history():
    with state_lock:
        return jsonify(chat_history)


@app.post("/api/chat/clear")
def clear_history():
    with state_lock:
        chat_history.clear()
    return jsonify(ok=True)


@app.post("/api/config")
def set_config():
    global rag_system
    data = _json_object()
    key = data.get("api_key") if data is not None else None
    if not isinstance(key, str) or not key.strip() or len(key) > 512:
        return jsonify(error="Введите API ключ"), 400
    # The UI key is a process-only override; it is never written to disk or sent
    # back through the API. Recreate the client with the complete configuration.
    with rag_lock:
        key = key.strip()
        if LLMConfig.from_env().api_key != key:
            rag_system = None
        os.environ["LLM_API_KEY"] = key
    return jsonify(ok=True)


@app.get("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
