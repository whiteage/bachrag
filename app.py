"""
app.py — Flask веб-интерфейс для RAG агента
"""

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

from cfg import LLMConfig
from newDocklingTest import process_pdf


# ─── Конфигурация ───────────────────────────────────────────────────────────

UPLOAD_DIR  = Path("uploads")
# Используем те же пути что и embedder.py — не создаём второй набор файлов
INDEX_PATH  = "brand.index"
DATA_PATH   = "data.pkl"
STATE_FILE  = "uploaded_files_state.json"

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# ─── Глобальное состояние ────────────────────────────────────────────────────

rag_system  = None
rag_lock    = threading.Lock()

indexed_files: dict[str, dict] = {}

indexing_progress: dict = {
    "active": False, "current": "", "done": 0, "total": 0, "log": []
}
# флаг: сколько файлов было done на момент последнего UI-опроса
_last_done_seen = 0

chat_history: list[dict] = []


# ─── Helpers ─────────────────────────────────────────────────────────────────

def save_files_state():
    try:
        state = {
            "files": {
                fname: {
                    "size":       meta.get("size", 0),
                    "status":     meta.get("status", "pending"),
                    "indexed_at": meta.get("indexed_at"),
                }
                for fname, meta in indexed_files.items()
            },
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_progress(f"⚠️ Ошибка сохранения состояния: {e}")


def load_files_state():
    global indexed_files
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
        for fname, meta in state.get("files", {}).items():
            fpath = UPLOAD_DIR / fname
            if fpath.exists():
                meta["size"] = fpath.stat().st_size
                indexed_files[fname] = meta
    except Exception as e:
        log_progress(f"⚠️ Ошибка загрузки состояния: {e}")


def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def fmt_size(n: int) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} ГБ"


def log_progress(msg: str):
    ts = time.strftime("%H:%M:%S")
    indexing_progress["log"].append(f"[{ts}] {msg}")
    indexing_progress["log"] = indexing_progress["log"][-200:]
    print(msg)


# ─── Загрузка RAG при старте ─────────────────────────────────────────────────

def try_load_rag() -> bool:
    global rag_system
    candidates = [
        (INDEX_PATH, DATA_PATH),   # brand.index + data.pkl
    ]
    for idx_path, data_path in candidates:
        if not (os.path.exists(idx_path) and os.path.exists(data_path)):
            continue
        try:
            index = faiss.read_index(idx_path)
            log_progress(f"✅ FAISS индекс: {index.ntotal} векторов ({idx_path})")

            if data_path.endswith(".json"):
                with open(data_path, encoding="utf-8") as f:
                    data = json.load(f)
            else:
                import pickle
                with open(data_path, "rb") as f:
                    data = pickle.load(f)
            log_progress(f"✅ Данные: {len(data)} чанков")

            from embedder import get_model, get_reranker
            model    = get_model()
            reranker = get_reranker()

            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            llm_cfg = LLMConfig(
                provider="deepseek", model="deepseek-chat",
                api_key=api_key, temperature=0.0, max_tokens=2000,
            )
            from test import SimpleLLMEnhancedRAG
            rag_system = SimpleLLMEnhancedRAG(
                index=index, data=data,
                embedding_model=model, reranker=reranker,
                llm_config=llm_cfg,
            )
            log_progress("🚀 RAG система готова!")
            return True
        except Exception as e:
            log_progress(f"❌ Ошибка загрузки {idx_path}: {e}")
    log_progress("⚠️ Существующий индекс не найден — нужно индексировать")
    return False


# ─── API: файлы ─────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"error": "Файлы не переданы"}), 400

    uploaded = []
    for file in request.files.getlist("files"):
        if not file.filename:
            continue
        name = secure_filename(file.filename)
        if not allowed(name):
            continue
        dest = UPLOAD_DIR / name
        file.save(dest)
        size = dest.stat().st_size
        # Если файл уже был проиндексирован — помечаем как modified
        prev = indexed_files.get(name, {})
        status = "modified" if prev.get("status") == "done" else "pending"
        indexed_files[name] = {"size": size, "status": status, "indexed_at": prev.get("indexed_at")}
        uploaded.append({"name": name, "size": fmt_size(size), "status": status})

    save_files_state()
    return jsonify({"uploaded": uploaded})


@app.route("/api/files", methods=["GET"])
def list_files():
    # Подбираем файлы с диска, которых нет в indexed_files
    for fpath in UPLOAD_DIR.iterdir():
        if fpath.is_file() and fpath.name not in indexed_files:
            indexed_files[fpath.name] = {
                "size": fpath.stat().st_size,
                "status": "pending",
                "indexed_at": None,
            }

    result = []
    for name, meta in list(indexed_files.items()):
        fpath = UPLOAD_DIR / name
        if not fpath.exists():
            # Файл удалён с диска — убираем из стейта
            indexed_files.pop(name, None)
            continue
        meta["size"] = fpath.stat().st_size
        result.append({
            "name":       name,
            "size":       fmt_size(meta["size"]),
            # BUG FIX: сервер хранит "done", UI ожидал "indexed" → унифицируем в "done"
            # CSS теперь тоже использует класс "done"
            "status":     meta["status"],
            "indexed_at": meta["indexed_at"],
        })

    order = {"pending": 0, "modified": 1, "indexing": 2, "done": 3, "error": 4}
    result.sort(key=lambda x: order.get(x["status"], 9))
    return jsonify(result)


@app.route("/api/files/<filename>", methods=["DELETE"])
def delete_file(filename: str):
    name = secure_filename(filename)
    path = UPLOAD_DIR / name
    if path.exists():
        path.unlink()
    indexed_files.pop(name, None)
    save_files_state()
    return jsonify({"ok": True})


# ─── API: индексирование ─────────────────────────────────────────────────────

def _save_index_and_data(index: faiss.Index, chunks: list[dict]):
    """Атомарно сохраняет FAISS индекс и pickle (формат embedder.py)."""
    import pickle
    tmp_idx  = INDEX_PATH + ".tmp"
    tmp_data = DATA_PATH  + ".tmp"
    faiss.write_index(index, tmp_idx)
    with open(tmp_data, "wb") as f:
        pickle.dump(chunks, f)
    os.replace(tmp_idx,  INDEX_PATH)
    os.replace(tmp_data, DATA_PATH)
    log_progress(f"💾 Сохранено: {index.ntotal} векторов, {len(chunks)} чанков → {INDEX_PATH}, {DATA_PATH}")


def _run_indexing():
    global rag_system

    indexing_progress["active"] = True
    indexing_progress["done"]   = 0
    indexing_progress["log"]    = []

    files = [f for f, m in indexed_files.items() if m["status"] in ("pending", "modified")]
    indexing_progress["total"] = len(files)

    if not files:
        indexing_progress["active"] = False
        return

    try:
        from embedder import get_model, get_reranker

        model    = get_model()
        reranker = get_reranker()

        # ── Загружаем существующий индекс и данные ────────────────────────
        import pickle
        if os.path.exists(INDEX_PATH) and os.path.exists(DATA_PATH):
            existing_index = faiss.read_index(INDEX_PATH)
            with open(DATA_PATH, "rb") as f:
                all_chunks: list[dict] = pickle.load(f)
            log_progress(f"📂 Загружен существующий индекс: {existing_index.ntotal} векторов, {len(all_chunks)} чанков")
        else:
            existing_index = None
            all_chunks     = []

        # ── Обрабатываем каждый новый файл ───────────────────────────────
        new_chunks: list[dict] = []

        for fname in files:
            indexing_progress["current"] = fname
            indexed_files[fname]["status"] = "indexing"
            log_progress(f"📄 Обработка: {fname}")

            try:
                docs = process_pdf(str(UPLOAD_DIR / fname))
                file_chunks = []
                for d in docs:
                    text = getattr(d, "page_content", None) or d.get("text", "")
                    if not text or len(text.strip()) < 10:
                        continue
                    file_chunks.append({
                        # BUG FIX: id = глобальный порядковый номер, уникален
                        "id":          len(all_chunks) + len(new_chunks),
                        "text":        text,
                        "metadata":    getattr(d, "metadata", {}),
                        "source_file": fname,
                    })
                new_chunks.extend(file_chunks)
                log_progress(f"✅ {fname}: {len(file_chunks)} чанков")
                indexed_files[fname]["status"]     = "done"
                indexed_files[fname]["indexed_at"] = time.strftime("%H:%M %d.%m")

            except Exception as e:
                log_progress(f"❌ Ошибка {fname}: {e}")
                indexed_files[fname]["status"] = "error"

            indexing_progress["done"] += 1

        if not new_chunks:
            log_progress("⚠️ Нет новых чанков для индексации")
            return

        # ── Строим эмбеддинги только для новых чанков ────────────────────
        texts = [c["text"] for c in new_chunks]
        log_progress(f"🔢 Создание эмбеддингов: {len(texts)} текстов...")

        new_embeddings = model.encode(
            texts,
            batch_size=8,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        faiss.normalize_L2(new_embeddings)

        # BUG FIX: создаём новый индекс = объединение старого + новые векторы.
        # IndexFlatIP поддерживает .add(), но после перезапуска нужно пересобрать
        # из сохранённых данных. Поэтому мы всегда пересобираем полный индекс —
        # так данные и индекс гарантированно синхронны.
        all_chunks.extend(new_chunks)

        all_texts = [c["text"] for c in all_chunks]
        log_progress(f"🔨 Пересборка полного индекса: {len(all_texts)} векторов...")

        all_embeddings = model.encode(
            all_texts,
            batch_size=8,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        faiss.normalize_L2(all_embeddings)

        full_index = faiss.IndexFlatIP(all_embeddings.shape[1])
        full_index.add(all_embeddings)

        # ── Сохраняем ─────────────────────────────────────────────────────
        _save_index_and_data(full_index, all_chunks)

        # ── Создаём RAG систему ───────────────────────────────────────────
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        llm_cfg = LLMConfig(
            provider="deepseek", model="deepseek-chat",
            api_key=api_key, temperature=0.0, max_tokens=2000,
        )
        from test import SimpleLLMEnhancedRAG
        with rag_lock:
            rag_system = SimpleLLMEnhancedRAG(
                index=full_index, data=all_chunks,
                embedding_model=model, reranker=reranker,
                llm_config=llm_cfg,
            )
        log_progress("🚀 RAG система обновлена и готова!")

    except Exception as e:
        import traceback
        log_progress(f"❌ Критическая ошибка: {e}")
        log_progress(traceback.format_exc())

    finally:
        indexing_progress["active"]  = False
        indexing_progress["current"] = ""
        save_files_state()


@app.route("/api/index", methods=["POST"])
def start_indexing():
    if indexing_progress["active"]:
        return jsonify({"error": "Индексирование уже запущено"}), 409
    pending = [f for f, m in indexed_files.items() if m["status"] in ("pending", "modified")]
    if not pending:
        return jsonify({"error": "Нет файлов для индексирования"}), 400
    threading.Thread(target=_run_indexing, daemon=True).start()
    return jsonify({"started": True, "files": pending})


@app.route("/api/index/status", methods=["GET"])
def index_status():
    return jsonify({
        **indexing_progress,
        "rag_ready": rag_system is not None,
    })


# ─── API: чат ────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    data  = request.json or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Пустой запрос"}), 400
    if rag_system is None:
        return jsonify({"error": "RAG не готов. Проиндексируйте документы."}), 503

    chat_history.append({
        "id": str(uuid.uuid4())[:8], "role": "user",
        "content": query, "ts": time.strftime("%H:%M"),
    })

    try:
        with rag_lock:
            result = rag_system.process_complex_query(query)

        answer  = result.get("synthesized_answer", "Ответ не найден.")
        sources = []
        for sq_results in result.get("results", {}).values():
            for r in sq_results[:2]:
                meta  = r.get("metadata", {})
                src   = meta.get("source") or meta.get("table") or "документ"
                page  = meta.get("page")
                label = Path(src).name if src else str(src)
                if page:
                    label += f" стр.{page}"
                if label not in sources:
                    sources.append(label)

        ai_msg = {
            "id":         str(uuid.uuid4())[:8],
            "role":       "assistant",
            "content":    answer,
            "sources":    sources[:6],
            "subqueries": [sq.get("subquery", "") for sq in result.get("subqueries", [])],
            "ts":         time.strftime("%H:%M"),
        }
        chat_history.append(ai_msg)
        return jsonify(ai_msg)

    except Exception as e:
        import traceback
        err = f"Ошибка: {e}"
        log_progress(f"❌ {err}\n{traceback.format_exc()}")
        return jsonify({"error": err}), 500


@app.route("/api/chat/history", methods=["GET"])
def get_history():
    return jsonify(chat_history[-100:])


@app.route("/api/chat/clear", methods=["POST"])
def clear_history():
    chat_history.clear()
    return jsonify({"ok": True})


@app.route("/api/config", methods=["POST"])
def set_config():
    data = request.json or {}
    if "api_key" in data:
        os.environ["DEEPSEEK_API_KEY"] = data["api_key"]
        # Обновляем ключ в существующем rag_system если он уже создан
        if rag_system is not None:
            try:
                rag_system.llm_client.client.api_key = data["api_key"]
            except Exception:
                pass
    return jsonify({"ok": True})


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


# ─── HTML ────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:      #0f0f11;
  --bg2:     #18181c;
  --bg3:     #222228;
  --bg4:     #2a2a32;
  --line:    rgba(255,255,255,0.07);
  --line2:   rgba(255,255,255,0.12);
  --txt:     #e8e8f0;
  --txt2:    #9090a8;
  --txt3:    #55556a;
  --accent:  #7b6ef6;
  --accent2: #a594f9;
  --green:   #4ade80;
  --amber:   #fbbf24;
  --red:     #f87171;
  --radius:  10px;
  --font:    'IBM Plex Sans', sans-serif;
  --mono:    'IBM Plex Mono', monospace;
}
html, body { height: 100%; overflow: hidden; background: var(--bg); color: var(--txt); font-family: var(--font); font-size: 14px; line-height: 1.6; }
.app { display: flex; height: 100vh; }

/* SIDEBAR */
.sidebar { width: 280px; min-width: 280px; background: var(--bg2); border-right: 1px solid var(--line); display: flex; flex-direction: column; }
.sidebar-top { padding: 20px 16px 0; }
.logo { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; }
.logo-icon { width: 32px; height: 32px; border-radius: 8px; background: var(--accent); display: flex; align-items: center; justify-content: center; }
.logo-icon svg { width: 18px; height: 18px; stroke: #fff; fill: none; }
.logo-name { font-size: 15px; font-weight: 500; }
.logo-sub  { font-size: 11px; color: var(--txt3); font-family: var(--mono); }
.api-row { margin-bottom: 16px; }
.api-row label { font-size: 10px; color: var(--txt3); text-transform: uppercase; letter-spacing: .08em; display: block; margin-bottom: 5px; }
.api-row input { width: 100%; background: var(--bg3); border: 1px solid var(--line2); border-radius: 6px; padding: 6px 10px; color: var(--txt); font-family: var(--mono); font-size: 11px; outline: none; }
.api-row input:focus { border-color: var(--accent); }
.upload-zone { margin: 0 0 12px; border: 1.5px dashed var(--line2); border-radius: var(--radius); padding: 18px; text-align: center; cursor: pointer; position: relative; transition: all .15s; }
.upload-zone:hover, .upload-zone.drag { border-color: var(--accent); background: rgba(123,110,246,.06); }
.upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.upload-zone svg { width: 24px; height: 24px; stroke: var(--txt3); fill: none; margin: 0 auto 6px; display: block; }
.upload-zone p   { font-size: 12px; color: var(--txt2); }
.upload-zone span  { color: var(--accent2); }
.upload-zone small { font-size: 10px; color: var(--txt3); display: block; margin-top: 3px; font-family: var(--mono); }
.section-label { font-size: 10px; color: var(--txt3); text-transform: uppercase; letter-spacing: .08em; padding: 0 16px 6px; }
.files-list { flex: 1; overflow-y: auto; padding: 4px 0; }
.files-list::-webkit-scrollbar { width: 4px; }
.files-list::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 2px; }
.file-item { display: flex; align-items: center; gap: 9px; padding: 7px 16px; cursor: default; }
.file-item:hover { background: var(--bg3); }
.file-ext { width: 30px; height: 30px; border-radius: 6px; background: var(--bg4); display: flex; align-items: center; justify-content: center; font-family: var(--mono); font-size: 9px; font-weight: 500; color: var(--txt2); text-transform: uppercase; flex-shrink: 0; }
.file-ext.pdf { background: rgba(248,113,113,.15); color: var(--red); }
.file-ext.txt { background: rgba(74,222,128,.12);  color: var(--green); }
.file-ext.md  { background: rgba(165,148,249,.15); color: var(--accent2); }
.file-ext.csv { background: rgba(251,191,36,.12);  color: var(--amber); }
.file-info { flex: 1; min-width: 0; }
.file-name { font-size: 12px; color: var(--txt); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.file-size { font-size: 10px; color: var(--txt3); font-family: var(--mono); }

/* BUG FIX: статус "done" вместо "indexed" — приводим CSS в соответствие с сервером */
.file-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.file-dot.pending  { background: var(--txt3); }
.file-dot.modified { background: var(--amber); }
.file-dot.indexing { background: var(--amber); animation: blink .8s infinite; }
.file-dot.done     { background: var(--green); }     /* ← было .indexed */
.file-dot.error    { background: var(--red); }

.file-del { width: 20px; height: 20px; border-radius: 4px; background: none; border: none; color: var(--txt3); cursor: pointer; display: flex; align-items: center; justify-content: center; opacity: 0; transition: opacity .1s; }
.file-item:hover .file-del { opacity: 1; }
.file-del:hover { color: var(--red); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.empty-files { padding: 24px 16px; text-align: center; font-size: 12px; color: var(--txt3); }
.sidebar-foot { padding: 12px 16px; border-top: 1px solid var(--line); }
.index-btn { width: 100%; padding: 9px; border-radius: var(--radius); border: 1px solid var(--line2); background: transparent; color: var(--txt2); font-family: var(--font); font-size: 12px; cursor: pointer; transition: all .15s; display: flex; align-items: center; justify-content: center; gap: 6px; }
.index-btn:hover:not(:disabled) { background: rgba(123,110,246,.12); border-color: var(--accent); color: var(--accent2); }
.index-btn:disabled { opacity: .35; cursor: not-allowed; }
.index-btn.running { background: rgba(251,191,36,.1); border-color: var(--amber); color: var(--amber); }
.spinner { width: 12px; height: 12px; border: 1.5px solid currentColor; border-top-color: transparent; border-radius: 50%; animation: spin .7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* CHAT */
.chat { flex: 1; display: flex; flex-direction: column; min-width: 0; background: var(--bg); }
.chat-header { display: flex; align-items: center; justify-content: space-between; padding: 0 24px; height: 56px; border-bottom: 1px solid var(--line); flex-shrink: 0; }
.rag-status { display: flex; align-items: center; gap: 6px; font-size: 11px; font-family: var(--mono); }
.status-dot { width: 6px; height: 6px; border-radius: 50%; }
.status-dot.ready   { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-dot.loading { background: var(--amber); animation: blink 1s infinite; }
.status-dot.offline { background: var(--txt3); }
.chat-actions { display: flex; gap: 6px; }
.btn-sm { padding: 5px 10px; border-radius: 6px; border: 1px solid var(--line2); background: transparent; color: var(--txt2); font-size: 11px; cursor: pointer; font-family: var(--font); transition: all .1s; }
.btn-sm:hover { background: var(--bg3); color: var(--txt); }
.messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; }
.messages::-webkit-scrollbar { width: 4px; }
.messages::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 2px; }
.welcome { margin: auto; text-align: center; max-width: 360px; }
.welcome-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 20px; }
.w-card { padding: 12px; background: var(--bg2); border: 1px solid var(--line); border-radius: var(--radius); cursor: pointer; text-align: left; transition: all .15s; }
.w-card:hover { border-color: var(--accent); background: rgba(123,110,246,.08); }
.w-card-icon { font-size: 16px; margin-bottom: 6px; }
.w-card-text { font-size: 11px; color: var(--txt2); line-height: 1.4; }
.msg { display: flex; gap: 12px; max-width: 820px; }
.msg.user { flex-direction: row-reverse; align-self: flex-end; }
.msg.user .bubble { background: rgba(123,110,246,.15); border: 1px solid rgba(123,110,246,.25); border-radius: 14px 4px 14px 14px; }
.msg.ai  .bubble { background: var(--bg2); border: 1px solid var(--line); border-radius: 4px 14px 14px 14px; }
.avatar { width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 500; }
.avatar.ai   { background: rgba(123,110,246,.2); color: var(--accent2); }
.avatar.user { background: var(--bg3); color: var(--txt2); }
.bubble { padding: 10px 14px; font-size: 13px; line-height: 1.65; color: var(--txt); max-width: 640px; }
.bubble pre  { background: var(--bg3); border-radius: 6px; padding: 10px; overflow-x: auto; font-size: 11px; font-family: var(--mono); margin: 8px 0; }
.bubble table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 12px; }
.bubble th, .bubble td { padding: 5px 10px; border: 1px solid var(--line2); text-align: left; }
.bubble th { background: var(--bg3); color: var(--txt2); }
.sources-row   { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px; }
.source-chip   { font-size: 10px; padding: 2px 8px; border-radius: 20px; border: 1px solid var(--line2); color: var(--txt3); font-family: var(--mono); }
.subqueries-row { margin-top: 6px; }
.subq-label  { font-size: 10px; color: var(--txt3); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
.subq-chips  { display: flex; flex-wrap: wrap; gap: 4px; }
.subq-chip   { font-size: 10px; padding: 2px 8px; border-radius: 4px; background: rgba(123,110,246,.1); color: var(--accent2); font-family: var(--mono); }
.msg-ts      { font-size: 10px; color: var(--txt3); margin-top: 4px; font-family: var(--mono); }
.thinking-dots { display: flex; gap: 4px; align-items: center; padding: 4px 0; }
.dot { width: 5px; height: 5px; border-radius: 50%; background: var(--txt3); animation: bounce 1.2s ease-in-out infinite; }
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}
.input-area { padding: 16px 24px 20px; border-top: 1px solid var(--line); flex-shrink: 0; }
.input-row  { display: flex; gap: 8px; align-items: flex-end; }
.input-box  { flex: 1; background: var(--bg2); border: 1px solid var(--line2); border-radius: var(--radius); overflow: hidden; transition: border-color .15s; }
.input-box:focus-within { border-color: var(--accent); }
.input-box textarea { width: 100%; padding: 10px 14px; background: transparent; border: none; outline: none; font-family: var(--font); font-size: 13px; color: var(--txt); resize: none; min-height: 42px; max-height: 140px; display: block; line-height: 1.5; }
.input-box textarea::placeholder { color: var(--txt3); }
.send-btn { width: 42px; height: 42px; border-radius: var(--radius); border: 1px solid var(--line2); background: transparent; color: var(--txt2); cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all .15s; }
.send-btn:hover:not(:disabled) { background: rgba(123,110,246,.15); border-color: var(--accent); color: var(--accent2); }
.send-btn:disabled { opacity: .25; cursor: not-allowed; }
.send-btn svg { width: 16px; height: 16px; }
.hint { font-size: 11px; color: var(--txt3); margin-top: 7px; font-family: var(--mono); }
.log-panel { border-top: 1px solid var(--line); padding: 12px 24px; max-height: 140px; overflow-y: auto; background: var(--bg2); font-family: var(--mono); font-size: 11px; color: var(--txt3); display: none; }
.log-panel.show { display: block; }
.log-panel::-webkit-scrollbar { width: 3px; }
.log-panel::-webkit-scrollbar-thumb { background: var(--bg4); }
.log-line     { padding: 1px 0; }
.log-line.ok  { color: var(--green); }
.log-line.err { color: var(--red); }
.bubble strong { font-weight: 500; color: var(--accent2); }
.bubble em     { font-style: italic; color: var(--txt2); }
.bubble ul, .bubble ol { padding-left: 18px; margin: 4px 0; }
.bubble li  { margin: 2px 0; }
.bubble h3, .bubble h4 { margin: 8px 0 4px; font-weight: 500; color: var(--txt2); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
</style>
</head>
<body>
<div class="app">

  <aside class="sidebar">
    <div class="sidebar-top">
      <div class="logo">
        <div class="logo-icon">
          <svg viewBox="0 0 24 24" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        </div>
        <div>
          <div class="logo-name">RAG Agent</div>
          <div class="logo-sub">DeepSeek · FAISS · Rerank</div>
        </div>
      </div>
      <div class="api-row">
        <label>DeepSeek API Key</label>
        <input type="password" id="apiKeyInput" placeholder="sk-..." oninput="saveApiKey(this.value)" />
      </div>
      <div class="upload-zone" id="dropZone">
        <input type="file" id="fileInput" multiple accept=".pdf,.txt,.md,.docx,.csv" onchange="handleFiles(this.files)" />
        <svg viewBox="0 0 24 24" stroke-width="1.5"><path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12M8 8l4-4 4 4" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <p><span>Загрузить файлы</span> или перетащить</p>
        <small>PDF · TXT · MD · DOCX · CSV</small>
      </div>
    </div>
    <div class="section-label">Документы</div>
    <div class="files-list" id="filesList">
      <div class="empty-files" id="emptyMsg">Нет загруженных файлов</div>
    </div>
    <div class="sidebar-foot">
      <button class="index-btn" id="indexBtn" disabled onclick="startIndexing()">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
        Индексировать
      </button>
    </div>
  </aside>

  <main class="chat">
    <div class="chat-header">
      <div class="rag-status">
        <div class="status-dot offline" id="statusDot"></div>
        <span id="statusText">Не готов</span>
      </div>
      <div class="chat-actions">
        <button class="btn-sm" onclick="toggleLog()">Лог</button>
        <button class="btn-sm" onclick="clearChat()">Очистить</button>
      </div>
    </div>

    <div class="messages" id="messages">
      <div class="welcome" id="welcome">
        <div style="font-size:36px;margin-bottom:12px">🔍</div>
        <div style="font-size:16px;font-weight:500;margin-bottom:6px">Поиск по документам</div>
        <div style="font-size:12px;color:var(--txt2)">Загрузите файлы, укажите API ключ и проиндексируйте базу знаний</div>
        <div class="welcome-grid">
          <div class="w-card" onclick="setQuery('Краткое содержание всех документов')"><div class="w-card-icon">📋</div><div class="w-card-text">Краткое содержание</div></div>
          <div class="w-card" onclick="setQuery('Какие ключевые показатели упомянуты?')"><div class="w-card-icon">📊</div><div class="w-card-text">Ключевые показатели</div></div>
          <div class="w-card" onclick="setQuery('Найди все упоминания дат и событий')"><div class="w-card-icon">📅</div><div class="w-card-text">Даты и события</div></div>
          <div class="w-card" onclick="setQuery('Сравни данные за разные периоды')"><div class="w-card-icon">📈</div><div class="w-card-text">Сравнение периодов</div></div>
        </div>
      </div>
    </div>

    <div class="log-panel" id="logPanel"></div>

    <div class="input-area">
      <div class="input-row">
        <div class="input-box">
          <textarea id="queryInput" placeholder="Задайте вопрос по документам..." rows="1"
            onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
        </div>
        <button class="send-btn" id="sendBtn" disabled onclick="sendQuery()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
      <div class="hint">↑ введите вопрос · Enter отправить · Shift+Enter перенос строки</div>
    </div>
  </main>
</div>

<script>
// ─── State ───────────────────────────────────────────────────────────────────
let ragReady         = false;
let isStreaming      = false;
let logVisible       = false;
// BUG FIX: инициализируем с active=false чтобы кнопка работала до первого pollStatus
let indexingProgress = { active: false, done: 0, total: 0, log: [] };
// BUG FIX: следим за флагом "только что завершилось" для однократного обновления
let wasIndexing      = false;
let cachedFiles      = [];  // последний известный список файлов

// ─── Init ────────────────────────────────────────────────────────────────────
// BUG FIX: один DOMContentLoaded, не два
document.addEventListener('DOMContentLoaded', () => {
  loadFiles();
  pollStatus();
  setInterval(pollStatus, 2000);
  autoResize(document.getElementById('queryInput'));
});

// ─── API Key ─────────────────────────────────────────────────────────────────
function saveApiKey(val) {
  if (val.length > 8)
    fetch('/api/config', { method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ api_key: val }) });
}

// ─── Drag & Drop ─────────────────────────────────────────────────────────────
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', ()  => dz.classList.remove('drag'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag');
  handleFiles(e.dataTransfer.files);
});

// ─── Upload ──────────────────────────────────────────────────────────────────
async function handleFiles(files) {
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append('files', f);
  try {
    const r    = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await r.json();
    if (data.uploaded?.length) await loadFiles();
  } catch(e) { console.error('Ошибка загрузки:', e); }
}

// ─── Files list ───────────────────────────────────────────────────────────────
async function loadFiles() {
  const r     = await fetch('/api/files');
  const files = await r.json();
  cachedFiles = files;
  renderFiles(files);
  updateIndexBtn(files);
}

function renderFiles(files) {
  const list  = document.getElementById('filesList');
  if (!files.length) {
    list.innerHTML = '<div class="empty-files">Нет загруженных файлов</div>';
    return;
  }
  list.innerHTML = files.map(f => {
    const ext = f.name.split('.').pop().toLowerCase();
    // Отображаемый статус для tooltip
    const statusLabel = { pending:'ожидает', modified:'изменён', indexing:'индексируется', done:'проиндексирован', error:'ошибка' }[f.status] || f.status;
    return `<div class="file-item">
      <div class="file-ext ${ext}">${ext}</div>
      <div class="file-info">
        <div class="file-name" title="${escHtml(f.name)}">${escHtml(f.name)}</div>
        <div class="file-size">${f.size}${f.indexed_at ? ' · ' + f.indexed_at : ''}</div>
      </div>
      <div class="file-dot ${f.status}" title="${statusLabel}"></div>
      <button class="file-del" onclick="deleteFile('${escHtml(f.name)}')" title="Удалить">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>`;
  }).join('');
}

async function deleteFile(name) {
  await fetch('/api/files/' + encodeURIComponent(name), { method: 'DELETE' });
  await loadFiles();
}

function updateIndexBtn(files) {
  const btn     = document.getElementById('indexBtn');
  const running = indexingProgress.active;
  // BUG FIX: учитываем и "modified"
  const pending = (files || []).filter(f => f.status === 'pending' || f.status === 'modified').length;

  if (running) {
    btn.disabled   = true;
    btn.className  = 'index-btn running';
    btn.innerHTML  = `<div class="spinner"></div> Индексирование...`;
  } else if (pending > 0) {
    btn.disabled   = false;
    btn.className  = 'index-btn';
    btn.innerHTML  = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Индексировать (${pending})`;
  } else {
    btn.disabled   = true;
    btn.className  = 'index-btn';
    btn.innerHTML  = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg> Индексировать`;
  }
}

// ─── Indexing ────────────────────────────────────────────────────────────────
async function startIndexing() {
  const key = document.getElementById('apiKeyInput').value.trim();
  if (key) saveApiKey(key);
  try { await fetch('/api/index', { method: 'POST' }); } catch(e) {}
}

async function pollStatus() {
  try {
    const r    = await fetch('/api/index/status');
    const data = await r.json();

    const wasActive = indexingProgress.active;
    indexingProgress = data;
    ragReady         = data.rag_ready;

    // Статус-точка
    const dot = document.getElementById('statusDot');
    const txt = document.getElementById('statusText');
    if (data.active) {
      dot.className = 'status-dot loading';
      txt.textContent = data.current ? `Обработка: ${data.current}` : 'Индексирование...';
    } else if (ragReady) {
      dot.className = 'status-dot ready';
      txt.textContent = 'RAG готов';
    } else {
      dot.className = 'status-dot offline';
      txt.textContent = 'Не готов';
    }

    document.getElementById('sendBtn').disabled = !ragReady || isStreaming;

    // BUG FIX: обновляем файлы ТОЛЬКО когда идёт индексирование
    // или только что завершилось (один раз, не каждый тик)
    if (data.active) {
      wasIndexing = true;
      await loadFiles();
    } else if (wasIndexing && !data.active) {
      // Только что завершилось — обновляем один раз
      wasIndexing = false;
      await loadFiles();
    } else {
      // Простой тик — используем кешированный список
      updateIndexBtn(cachedFiles);
    }

    if (data.log?.length) renderLog(data.log);
  } catch(e) {}
}

function renderLog(lines) {
  const panel = document.getElementById('logPanel');
  panel.innerHTML = lines.map(l => {
    const cls = l.includes('✅') || l.includes('готов') ? 'ok'
              : l.includes('❌') || l.includes('Ошибка') ? 'err' : '';
    return `<div class="log-line ${cls}">${escHtml(l)}</div>`;
  }).join('');
  panel.scrollTop = panel.scrollHeight;
}

function toggleLog() {
  logVisible = !logVisible;
  document.getElementById('logPanel').classList.toggle('show', logVisible);
}

// ─── Chat ────────────────────────────────────────────────────────────────────
function setQuery(text) {
  const inp = document.getElementById('queryInput');
  inp.value = text;
  autoResize(inp);
  inp.focus();
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

async function sendQuery() {
  const inp = document.getElementById('queryInput');
  const q   = inp.value.trim();
  if (!q || !ragReady || isStreaming) return;

  inp.value   = '';
  autoResize(inp);
  isStreaming  = true;
  document.getElementById('sendBtn').disabled = true;
  hideWelcome();

  appendMsg({ role: 'user', content: q, ts: timestamp() });

  const msgs    = document.getElementById('messages');
  const thinkEl = document.createElement('div');
  thinkEl.className = 'msg ai';
  thinkEl.innerHTML = `<div class="avatar ai">AI</div><div><div class="bubble"><div class="thinking-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div></div>`;
  msgs.appendChild(thinkEl);
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const r    = await fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ query: q }),
    });
    const data = await r.json();
    thinkEl.remove();
    appendMsg(data.error ? { role: 'ai', content: '❌ ' + data.error, ts: timestamp() } : data);
  } catch(e) {
    thinkEl.remove();
    appendMsg({ role: 'ai', content: '❌ Ошибка соединения с сервером', ts: timestamp() });
  }

  isStreaming = false;
  document.getElementById('sendBtn').disabled = !ragReady;
}

function appendMsg(msg) {
  const msgs   = document.getElementById('messages');
  const isUser = msg.role === 'user';
  const div    = document.createElement('div');
  div.className = `msg ${isUser ? 'user' : 'ai'}`;

  const contentHtml = isUser ? escHtml(msg.content) : renderMarkdown(msg.content);

  let extras = '';
  if (!isUser && msg.subqueries?.length)
    extras += `<div class="subqueries-row"><div class="subq-label">Подзапросы</div><div class="subq-chips">${msg.subqueries.map(s=>`<span class="subq-chip">${escHtml(s)}</span>`).join('')}</div></div>`;
  if (!isUser && msg.sources?.length)
    extras += `<div class="sources-row">${msg.sources.map(s=>`<span class="source-chip">📄 ${escHtml(s)}</span>`).join('')}</div>`;

  div.innerHTML = `
    <div class="avatar ${isUser ? 'user' : 'ai'}">${isUser ? 'Вы' : 'AI'}</div>
    <div>
      <div class="bubble">${contentHtml}${extras}</div>
      <div class="msg-ts">${msg.ts || ''}</div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

async function clearChat() {
  await fetch('/api/chat/clear', { method: 'POST' });
  document.getElementById('messages').innerHTML = '';
  showWelcome();
}

function hideWelcome() { document.getElementById('welcome')?.remove(); }

function showWelcome() {
  const msgs = document.getElementById('messages');
  const w    = document.createElement('div');
  w.id        = 'welcome';
  w.className = 'welcome';
  w.innerHTML = `
    <div style="font-size:36px;margin-bottom:12px">🔍</div>
    <div style="font-size:16px;font-weight:500;margin-bottom:6px">Поиск по документам</div>
    <div style="font-size:12px;color:var(--txt2)">Задайте вопрос по загруженным документам</div>
    <div class="welcome-grid">
      <div class="w-card" onclick="setQuery('Краткое содержание всех документов')"><div class="w-card-icon">📋</div><div class="w-card-text">Краткое содержание</div></div>
      <div class="w-card" onclick="setQuery('Ключевые показатели в документах')"><div class="w-card-icon">📊</div><div class="w-card-text">Ключевые показатели</div></div>
    </div>`;
  msgs.appendChild(w);
}

function renderMarkdown(text) {
  if (!text) return '';
  let h = escHtml(text);
  h = h.replace(/```([\s\S]*?)```/g, (_, c) => `<pre>${c}</pre>`);
  h = h.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  h = h.replace(/^### (.+)$/gm,  '<h3>$1</h3>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g,     '<em>$1</em>');
  h = h.replace(/((\|.+\|\n)+)/g, m => {
    const rows = m.trim().split('\n').filter(r => !/^\|[-| :]+\|$/.test(r));
    return '<table>' + rows.map((r, i) => {
      const cells = r.split('|').filter(c => c.trim());
      const tag   = i === 0 ? 'th' : 'td';
      return `<tr>${cells.map(c => `<${tag}>${c.trim()}</${tag}>`).join('')}</tr>`;
    }).join('') + '</table>';
  });
  h = h.replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>');
  h = h.replace(/(<li>[\s\S]*?<\/li>\n?)+/g, m => `<ul>${m}</ul>`);
  h = h.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
  h = h.replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>');
  return h;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function timestamp() {
  return new Date().toLocaleTimeString('ru', { hour:'2-digit', minute:'2-digit' });
}
</script>
</body>
</html>"""

# ─── Запуск ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv()
    UPLOAD_DIR.mkdir(exist_ok=True)

    load_files_state()

    # Подбираем файлы с диска, которых нет в состоянии
    for fpath in UPLOAD_DIR.iterdir():
        if fpath.is_file() and fpath.name not in indexed_files:
            indexed_files[fpath.name] = {
                "size": fpath.stat().st_size,
                "status": "pending",
                "indexed_at": None,
            }
    save_files_state()

    try_load_rag()

    print(f"📂 Файлов в базе: {len(indexed_files)}")
    print("🚀 RAG UI: http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)