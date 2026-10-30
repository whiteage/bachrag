"""Embedding models, index persistence and two-stage document retrieval."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from cfg import settings

if TYPE_CHECKING:
    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

INDEX_PATH = str(settings.index_path)
DATA_PATH = str(settings.data_path)
INSTRUCT_PROMPT = (
    "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
)


def get_model(device: str | None = None) -> SentenceTransformer:
    """Load the embedding model only when indexing or retrieval needs it."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.startswith("cuda") and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = SentenceTransformer(
        "ai-sage/Giga-Embeddings-instruct",
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": dtype},
    )
    model.max_seq_length = 4096
    return model


def get_reranker(device: str | None = None) -> CrossEncoder:
    import torch
    from sentence_transformers import CrossEncoder

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=512)


def _normalize_chunks(data: list[dict]) -> list[dict]:
    if not isinstance(data, list):
        raise ValueError("Данные индекса должны быть списком фрагментов")
    normalized = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Некорректный фрагмент {idx}")
        text = item.get("text", item.get("page_content"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"У фрагмента {idx} отсутствует текст")
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Некорректные метаданные фрагмента {idx}")
        normalized.append({**item, "id": idx, "text": text, "metadata": metadata})
    return normalized


def load_chunks_from_json(json_path: str | Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as stream:
        return _normalize_chunks(json.load(stream))


def build_index(json_path: str | Path, model: SentenceTransformer,
                batch_size: int = 4) -> tuple[faiss.Index, list[dict]]:
    import faiss

    data = load_chunks_from_json(json_path)
    if not data:
        raise ValueError("Нельзя построить индекс без фрагментов")
    embeddings = model.encode(
        [item["text"] for item in data], batch_size=batch_size,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, data


def save_all(index: faiss.Index, data: list[dict]) -> None:
    """Persist local index data as JSON, replacing each file atomically."""
    import faiss

    data = _normalize_chunks(data)
    if index.ntotal != len(data):
        raise ValueError("Количество векторов не соответствует числу фрагментов")
    index_path, data_path = Path(INDEX_PATH), Path(DATA_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        for target in (index_path, data_path):
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary_paths.append(Path(stream.name))
        faiss.write_index(index, str(temporary_paths[0]))
        with temporary_paths[1].open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        os.replace(temporary_paths[0], index_path)
        os.replace(temporary_paths[1], data_path)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def load_index_data() -> tuple[faiss.Index, list[dict]]:
    import faiss

    data = load_chunks_from_json(DATA_PATH)
    index = faiss.read_index(INDEX_PATH)
    if index.ntotal != len(data):
        raise ValueError(f"Несоответствие: индекс={index.ntotal}, данные={len(data)}")
    return index, data


def load_all() -> tuple[faiss.Index, list[dict], SentenceTransformer, CrossEncoder]:
    index, data = load_index_data()
    return index, data, get_model(), get_reranker()


def search_simple(query: str, index: faiss.Index, data: list[dict],
                  model: SentenceTransformer, k: int = 5) -> list[dict]:
    """Retrieve valid FAISS rows without mutating stored chunks."""
    import faiss

    if not query.strip() or k <= 0 or not data or index.ntotal == 0:
        return []
    query_vector = model.encode(
        [query], prompt=INSTRUCT_PROMPT, convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, min(k, index.ntotal))
    results = []
    seen = set()
    for score, row in zip(distances[0], indices[0]):
        row = int(row)
        if 0 <= row < len(data) and row not in seen:
            seen.add(row)
            results.append({**data[row], "embedding_score": float(score)})
    return results


def search_with_rerank(query: str, index: faiss.Index, data: list[dict],
                       model: SentenceTransformer, reranker: CrossEncoder,
                       initial_k: int = 20, final_k: int = 5) -> list[dict]:
    if final_k <= 0:
        return []
    candidates = search_simple(query, index, data, model, k=initial_k)
    if not candidates:
        return []
    # CrossEncoder handles token truncation; cutting characters would lose evidence.
    scores = reranker.predict([[query, doc["text"]] for doc in candidates], batch_size=8)
    if len(scores) != len(candidates):
        raise ValueError("Реранкер вернул неверное число оценок")
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
    candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
    return candidates[:final_k]


def enhanced_search_with_rerank(query: str, index: faiss.Index, data: list[dict],
                                model: SentenceTransformer, reranker: CrossEncoder,
                                initial_k: int = 30, final_k: int = 5,
                                score_threshold: float = 0.1) -> list[dict]:
    candidates = search_with_rerank(
        query, index, data, model, reranker, initial_k=initial_k, final_k=final_k,
    )
    filtered = [item for item in candidates if item["rerank_score"] >= score_threshold]
    return filtered or candidates[:1]
