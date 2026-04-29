import json
import pickle
import re
from typing import List, Tuple
import os
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

INDEX_PATH = "brand.index"
DATA_PATH = "data.pkl"

INSTRUCT_PROMPT = (
    "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
)


# =========================
# 1. Загрузка моделей
# =========================

def get_model(device: str | None = None) -> SentenceTransformer:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        "ai-sage/Giga-Embeddings-instruct",
        device=device,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": "True"
        },
        config_kwargs={
            "trust_remote_code": "True"
        }
    )
    model.max_seq_length = 4096
    # Проверка
    dim = len(model.encode(["тест"], show_progress_bar=False)[0])
    print(f"✅ Модель эмбеддингов загружена. Устройство: {device}. Размерность: {dim}")
    return model


def get_reranker(device: str | None = None) -> CrossEncoder:
    """Загружаем реранкер BGE"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "BAAI/bge-reranker-v2-m3"
    reranker = CrossEncoder(
        model_name,
        device=device,
        max_length=512,  # реранкеру не нужно 4096, 512 достаточно
    )
    print(f"✅ Реранкер загружен: {model_name} на {device}")
    return reranker


# =========================
# 2. Загрузка данных
# =========================

def load_chunks_from_json(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for idx, item in enumerate(data):
        item["id"] = idx
        item["text"] = item["page_content"]
    print(f"✅ Загружено {len(data)} чанков из {json_path}")
    return data


# =========================
# 3. Построение индекса
# =========================

def build_index(json_path: str, model: SentenceTransformer,
                batch_size: int = 4) -> tuple[faiss.Index, list[dict]]:
    data = load_chunks_from_json(json_path)
    texts = [x["text"] for x in data]
    print("Создание эмбеддингов...")
    # документы — без instruct-промпта
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"✅ Индекс построен. Векторов: {index.ntotal}")
    return index, data


# =========================
# 4. Сохранение / загрузка
# =========================

def save_all(index: faiss.Index, data: list[dict]):
    faiss.write_index(index, INDEX_PATH)
    with open(DATA_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Сохранено → {INDEX_PATH}, {DATA_PATH}")


def load_all() -> tuple[faiss.Index, list[dict], SentenceTransformer, CrossEncoder]:
    index = faiss.read_index(INDEX_PATH)
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    if index.ntotal != len(data):
        raise ValueError(f"Несоответствие: индекс={index.ntotal}, данные={len(data)}")
    model = get_model()
    reranker = get_reranker()
    print(f"✅ Загружено: {index.ntotal} векторов")
    return index, data, model, reranker


# =========================
# 5. Поиск (двухэтапный с реранкером)
# =========================

def search_with_rerank(
        query: str,
        index: faiss.Index,
        data: list[dict],
        model: SentenceTransformer,
        reranker: CrossEncoder,
        initial_k: int = 20,  # сколько достаем из FAISS
        final_k: int = 5  # сколько оставляем после реранкера
) -> list[dict]:
    """
    Двухэтапный поиск:
    1. Быстрый поиск по эмбеддингам (initial_k кандидатов)
    2. Точный реранкинг кросс-энкодером (оставляем final_k)
    """
    # Этап 1: Быстрый поиск
    query_vec = model.encode(
        [query],
        prompt=INSTRUCT_PROMPT,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, initial_k)

    # Собираем кандидатов
    candidates = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        candidates.append({
            "embedding_score": float(score),
            **data[idx]
        })

    if not candidates:
        return []

    # Этап 2: Реранкинг
    # Подготавливаем пары [query, document_text]
    pairs = [[query, doc["text"]] for doc in candidates]

    # Получаем скоры релевантности от кросс-энкодера
    rerank_scores = reranker.predict(pairs)

    # Добавляем скоры к кандидатам и сортируем
    for doc, rerank_score in zip(candidates, rerank_scores):
        doc["rerank_score"] = float(rerank_score)

    # Сортируем по убыванию rerank_score
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Оставляем top final_k
    return candidates[:final_k]


def search_simple(
        query: str,
        index: faiss.Index,
        data: list[dict],
        model: SentenceTransformer,
        k: int = 5
) -> list[dict]:
    """Обычный поиск без реранкера (для сравнения)"""
    query_vec = model.encode(
        [query],
        prompt=INSTRUCT_PROMPT,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({"embedding_score": float(score), **data[idx]})
    return results


# =========================
# 6. Тест поиска с реранкером
# =========================

def test_search_with_rerank(
        index,
        data,
        model,
        reranker,
        queries: list[str],
        initial_k: int = 20,
        final_k: int = 5
):
    for query in queries:
        print(f"\n{'=' * 80}")
        print(f"Запрос: {query}")
        print('=' * 80)

        # Поиск с реранкером
        print("\n🔍 РЕЗУЛЬТАТЫ С РЕРАНКЕРОМ (Cross-Encoder):")
        print("-" * 80)
        results_reranked = search_with_rerank(
            query, index, data, model, reranker,
            initial_k=initial_k, final_k=final_k
        )

        for i, r in enumerate(results_reranked, 1):
            meta = r["metadata"]
            print(f"\n{i}. Rerank Score: {r['rerank_score']:.4f} | Embedding Score: {r['embedding_score']:.4f}")
            print(f"   Стр.{meta.get('page', 'N/A')} | Тип: {meta.get('type', 'N/A')}")
            print(f"   Текст: {r['text'][:200]}...")

        # Для сравнения - обычный поиск
        print("\n📊 ДЛЯ СРАВНЕНИЯ - ОБЫЧНЫЙ ПОИСК (только эмбеддинги):")
        print("-" * 80)
        results_simple = search_simple(query, index, data, model, k=final_k)

        for i, r in enumerate(results_simple, 1):
            meta = r["metadata"]
            print(f"\n{i}. Embedding Score: {r['embedding_score']:.4f}")
            print(f"   Стр.{meta.get('page', 'N/A')} | Тип: {meta.get('type', 'N/A')}")
            print(f"   Текст: {r['text'][:200]}...")


# =========================
# 7. Функция для RAG пайплайна
# =========================

def get_context_for_llm(
        query: str,
        index: faiss.Index,
        data: list[dict],
        model: SentenceTransformer,
        reranker: CrossEncoder,
        initial_k: int = 20,
        final_k: int = 3
) -> str:
    """
    Возвращает готовый контекст для подстановки в промпт LLM
    """
    results = search_with_rerank(
        query, index, data, model, reranker,
        initial_k=initial_k, final_k=final_k
    )

    if not results:
        return "Релевантные документы не найдены."

    context_parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        context_parts.append(
            f"[Документ {i}] (стр.{meta.get('page', 'N/A')}, релевантность: {r['rerank_score']:.3f})\n"
            f"{r['text']}\n"
        )

    return "\n".join(context_parts)


def enhanced_search_with_rerank(
        query: str,
        index: faiss.Index,
        data: list[dict],
        model: SentenceTransformer,
        reranker: CrossEncoder,
        initial_k: int = 30,  # Увеличим для лучшего покрытия
        final_k: int = 5,
        score_threshold: float = 0.1  # Отсекаем явно нерелевантные
) -> List[dict]:
    """
    Улучшенный двухэтапный поиск с фильтрацией по порогу
    """
    # Этап 1: Быстрый поиск
    query_vec = model.encode(
        [query],
        prompt=INSTRUCT_PROMPT,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, initial_k)

    # Собираем кандидатов
    candidates = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        candidates.append({
            "embedding_score": float(score),
            **data[idx]
        })

    if not candidates:
        return []

    # Этап 2: Реранкинг с батчированием для скорости
    pairs = [[query, doc["text"][:512]] for doc in candidates]  # Обрезаем длинные тексты

    # Получаем скоры
    rerank_scores = reranker.predict(pairs, batch_size=8)

    # Добавляем скоры и фильтруем
    for doc, rerank_score in zip(candidates, rerank_scores):
        doc["rerank_score"] = float(rerank_score)

    # Сортируем и фильтруем
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Оставляем только те, что выше порога
    filtered = [c for c in candidates if c["rerank_score"] >= score_threshold]

    # Если после фильтрации ничего не осталось, берем топ-1
    if not filtered:
        filtered = candidates[:1]

    return filtered[:final_k]


def extract_financial_numbers(text: str) -> List[Tuple[str, str]]:
    """
    Извлекает финансовые показатели из текста для лучшего контекста
    """
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*(?:млрд|миллиард)', 'млрд руб'),
        (r'(\d+(?:\.\d+)?)\s*(?:млн|миллион)', 'млн руб'),
        (r'(\d+(?:\.\d+)?)\s*%', '%'),
    ]

    results = []
    for pattern, unit in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches[:3]:  # Берем первые 3 совпадения
            results.append((match, unit))

    return results


def format_context_for_llm(
        query: str,
        results: List[dict],
        include_metadata: bool = True
) -> str:
    """
    Форматирует контекст для LLM с выделением ключевой информации
    """
    if not results:
        return "Релевантные документы не найдены."

    context_parts = []

    # Определяем тип запроса для лучшего форматирования
    is_financial = any(word in query.lower() for word in
                       ['финансов', 'показател', 'цифр', 'млрд', 'доход', 'прибыль'])

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        text = r["text"]

        # Базовая информация
        part = f"[Источник {i}] "
        if include_metadata:
            part += f"(стр.{meta.get('page', 'N/A')}, "
            part += f"релевантность: {r['rerank_score']:.3f})\n"

        # Для финансовых запросов выделяем числа
        if is_financial:
            numbers = extract_financial_numbers(text)
            if numbers:
                part += "💰 Ключевые показатели:\n"
                for num, unit in numbers:
                    part += f"   • {num} {unit}\n"
                part += "\n"

        part += f"{text}\n"
        part += "-" * 40 + "\n"

        context_parts.append(part)

    return "\n".join(context_parts)


# =========================
# Функция для интерактивного тестирования
# =========================

def interactive_search(
        index: faiss.Index,
        data: list[dict],
        model: SentenceTransformer,
        reranker: CrossEncoder
):
    """
    Интерактивный режим поиска
    """
    print("\n" + "🔍" * 40)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ ПОИСКА")
    print("🔍" * 40)
    print("Введите 'exit' для выхода, 'compare' для сравнения методов\n")

    while True:
        query = input("\n🤔 Ваш вопрос: ").strip()

        if query.lower() == 'exit':
            break

        if query.lower() == 'compare':
            query = input("Вопрос для сравнения: ").strip()
            print("\n📊 СРАВНЕНИЕ МЕТОДОВ:")

            # Обычный поиск
            simple_results = search_simple(query, index, data, model, k=3)
            print("\nБез реранкера (только эмбеддинги):")
            for i, r in enumerate(simple_results, 1):
                print(f"{i}. Score: {r['embedding_score']:.4f} | {r['text'][:100]}...")

            # С реранкером
            reranked = enhanced_search_with_rerank(
                query, index, data, model, reranker,
                initial_k=20, final_k=3
            )
            print("\nС реранкером (Cross-Encoder):")
            for i, r in enumerate(reranked, 1):
                print(f"{i}. Rerank: {r['rerank_score']:.4f} | {r['text'][:100]}...")

            continue

        # Обычный поиск
        results = enhanced_search_with_rerank(
            query, index, data, model, reranker,
            initial_k=30, final_k=3
        )

        if not results:
            print("❌ Ничего не найдено")
            continue

        print(f"\n✅ Найдено {len(results)} релевантных фрагментов:\n")

        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"{i}. [Score: {r['rerank_score']:.4f}] Стр.{meta.get('page', 'N/A')}")
            print(f"   {r['text'][:200]}...")
            print()


# =========================
# 8. Main
# =========================

# =========================
# 8. Main
# =========================

if __name__ == "__main__":

    # Проверяем наличие файлов
    if os.path.exists(INDEX_PATH) and os.path.exists(DATA_PATH):
        print("📂 Загружаем существующий индекс...")
        index, data, model, reranker = load_all()
    else:
        print("🔨 Создаем новый индекс...")
        model = get_model()
        reranker = get_reranker()

        json_path = "обобщенный отчет сбер за 2025_chunks4.json"
        if not os.path.exists(json_path):
            print(f"❌ Файл {json_path} не найден!")
            exit(1)

        index, data = build_index(json_path, model)
        save_all(index, data)

    # Демонстрация улучшенного форматирования
    print("\n" + "=" * 80)
    print("📈 ПРИМЕР УЛУЧШЕННОГО КОНТЕКСТА ДЛЯ LLM:")
    print("=" * 80)

    query = "Какие финансовые показатели у Сбера в 2025 году?"
    results = enhanced_search_with_rerank(
        query, index, data, model, reranker,
        initial_k=30, final_k=3, score_threshold=0.1
    )

    context = format_context_for_llm(query, results)
    print(context)

    # Запускаем интерактивный режим
    interactive_search(index, data, model, reranker)