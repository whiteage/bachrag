import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import faiss
from openai import OpenAI  # или любой другой клиент
from sentence_transformers import SentenceTransformer, CrossEncoder

from embedder import load_all, INDEX_PATH, DATA_PATH, enhanced_search_with_rerank




class LLMClient:
    """Универсальный клиент для работы с LLM"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.base_url
        )

    def complete(self, prompt: str, system_prompt: str = None) -> str:
        """Базовый вызов LLM"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            response_format={"type": "json_object"}  # заставляем вернуть JSON
        )

        return response.choices[0].message.content


# =========================
# Шаг 1: Декомпозиция запроса через LLM
# =========================

class QueryDecomposer:
    """Разбивает сложный запрос на подзапросы с помощью LLM"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def decompose(self, query: str) -> List[Dict[str, Any]]:
        """
        Возвращает список подзапросов с метаданными
        """
        system_prompt = """Ты - система анализа поисковых запросов. 
Твоя задача - разбить сложный запрос на простые подзапросы для поиска в финансовых документах.
Для каждого подзапроса укажи:
- subquery: текст подзапроса
- entity_type: тип сущности (капитал, облигации, прибыль, активы, и т.д.)
- year: год, если указан (или null)
- priority: приоритет от 1 до 10
- search_k: сколько результатов нужно найти (рекомендуемое)

Верни ответ в формате JSON."""

        prompt = f"""Разбей запрос на подзапросы для эффективного поиска:

Запрос: "{query}"

Пример ответа для запроса "капитал за 2024 и 2025, облигации":
{{
    "original_query": "капитал за 2024 и 2025, облигации",
    "subqueries": [
        {{
            "subquery": "капитал 2024",
            "entity_type": "капитал",
            "year": 2024,
            "priority": 9,
            "search_k": 3
        }},
        {{
            "subquery": "капитал 2025",
            "entity_type": "капитал",
            "year": 2025,
            "priority": 9,
            "search_k": 3
        }},
        {{
            "subquery": "облигации портфель",
            "entity_type": "облигации",
            "year": null,
            "priority": 7,
            "search_k": 2
        }}
    ]
}}

Верни JSON с разбивкой для запроса: "{query}"
"""

        response = self.llm.complete(prompt, system_prompt)
        try:
            result = json.loads(response)
            return result.get("subqueries", [])
        except:
            # Fallback: простое разбиение
            return [{"subquery": query, "entity_type": "general", "year": None, "priority": 10, "search_k": 5}]


# =========================
# Шаг 2: Агрегация и синтез через LLM
# =========================

class ResultAggregator:
    """Объединяет результаты подзапросов и формирует финальный ответ"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def aggregate_and_synthesize(
            self,
            original_query: str,
            subqueries_results: Dict[str, List[Dict]],
            max_context_length: int = 4000
    ) -> str:
        """
        Синтезирует ответ из результатов всех подзапросов
        """
        # Подготавливаем контекст из найденных чанков
        context_parts = []

        for subquery, results in subqueries_results.items():
            if not results:
                continue

            context_parts.append(f"\n=== Результаты по подзапросу: '{subquery}' ===\n")

            for i, r in enumerate(results[:3], 1):  # Берем топ-3 по каждому подзапросу
                meta = r.get("metadata", {})
                text = r.get("text", "")
                score = r.get("rerank_score", 0)

                context_parts.append(
                    f"[Источник {i}, стр.{meta.get('page', 'N/A')}, "
                    f"релевантность: {score:.3f}]\n{text}\n"
                )

        context = "\n".join(context_parts)

        # Обрезаем контекст если слишком длинный
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        # Формируем промпт для синтеза
        system_prompt = """Ты -  аналитик. Твоя задача - синтезировать точный и структурированный ответ 
на основе найденных фрагментов документов. 

Правила:
1. Отвечай только на основе предоставленного контекста
2. Если информации недостаточно - честно скажи об этом
3. Для чисел и дат обязательно указывай источники (номера страниц)
4. Структурируй ответ: сначала сводка, потом детали
5. Используй таблицы для сравнения данных по годам"""

        prompt = f"""На основе следующих фрагментов из книги/отчета/методички, ответь на вопрос.

ВОПРОС: {original_query}

НАЙДЕННЫЕ ФРАГМЕНТЫ:
{context}

Синтезируй структурированный ответ. Если есть данные за разные годы - представь их в виде сравнения.
"""

        response = self.llm.complete(prompt, system_prompt)
        return response


# =========================
# Шаг 3: Умный выбор чанков для LLM (Relevance Scoring)
# =========================

class ChunkSelector:
    """Выбирает наиболее релевантные чанки с помощью LLM"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def select_best_chunks(
            self,
            query: str,
            candidates: List[Dict],
            max_chunks: int = 5
    ) -> List[Dict]:
        """
        Использует LLM для выбора самых релевантных чанков из кандидатов
        """
        if len(candidates) <= max_chunks:
            return candidates

        # Подготавливаем кандидатов для оценки
        chunks_text = []
        for i, chunk in enumerate(candidates):
            text = chunk.get("text", "")[:300]  # Превью для LLM
            chunks_text.append(f"[{i}] {text}")

        system_prompt = """Ты - система оценки релевантности. 
Оцени какие фрагменты наиболее полно отвечают на запрос.
Верни JSON с массивом индексов лучших фрагментов."""

        prompt = f"""Запрос: "{query}"

Фрагменты для оценки:
{chr(10).join(chunks_text[:15])}  # Оцениваем только первые 15

Выбери {max_chunks} наиболее релевантных фрагментов. 
Верни JSON в формате: {{"selected_indices": [0, 3, 5]}}
"""

        response = self.llm.complete(prompt, system_prompt)

        try:
            data = json.loads(response)
            selected = [candidates[i] for i in data["selected_indices"] if i < len(candidates)]
            return selected[:max_chunks]
        except:
            # Fallback: берем топ по скорам
            return candidates[:max_chunks]


# =========================
# Основной класс RAG с LLM-управлением
# =========================

class LLMEnhancedRAG:
    """RAG система с управлением через LLM"""

    def __init__(
            self,
            index: faiss.Index,
            data: List[dict],
            embedding_model: SentenceTransformer,
            reranker: CrossEncoder,
            llm_config: LLMConfig
    ):
        self.index = index
        self.data = data
        self.embedding_model = embedding_model
        self.reranker = reranker

        self.llm_client = LLMClient(llm_config)
        self.decomposer = QueryDecomposer(self.llm_client)
        self.aggregator = ResultAggregator(self.llm_client)
        self.selector = ChunkSelector(self.llm_client)

    def search_subquery(
            self,
            subquery: str,
            k: int = 5
    ) -> List[Dict]:
        """Поиск по одному подзапросу"""
        return enhanced_search_with_rerank(
            subquery,
            self.index,
            self.data,
            self.embedding_model,
            self.reranker,
            initial_k=30,
            final_k=k,
            score_threshold=0.1
        )

    def process_complex_query(
            self,
            query: str,
            return_synthesized: bool = True
    ) -> Dict[str, Any]:
        """
        Полный цикл обработки сложного запроса
        """
        print(f"🤔 Обработка запроса: {query}")

        # Шаг 1: Декомпозиция через LLM
        print("📋 Декомпозиция запроса...")
        subqueries = self.decomposer.decompose(query)

        # Шаг 2: Параллельный поиск по всем подзапросам
        print(f"🔍 Поиск по {len(subqueries)} подзапросам...")
        all_results = {}

        for sq in subqueries:
            subquery_text = sq["subquery"]
            k = sq.get("search_k", 3)

            print(f"  - {subquery_text} (k={k})")
            results = self.search_subquery(subquery_text, k=k)

            if results:
                all_results[subquery_text] = results

        # Шаг 3: Выбор лучших чанков через LLM (опционально)
        if len(all_results) > 1:
            print("🎯 Выбор наиболее релевантных чанков...")
            all_candidates = []
            for results in all_results.values():
                all_candidates.extend(results)

            # Убираем дубликаты
            seen = set()
            unique_candidates = []
            for c in all_candidates:
                if c["id"] not in seen:
                    seen.add(c["id"])
                    unique_candidates.append(c)

            # LLM выбирает лучшие
            best_chunks = self.selector.select_best_chunks(query, unique_candidates, max_chunks=8)

            # Перегруппируем
            filtered_results = {}
            for sq, results in all_results.items():
                filtered = [r for r in results if r in best_chunks]
                if filtered:
                    filtered_results[sq] = filtered
            all_results = filtered_results

        # Шаг 4: Синтез ответа через LLM
        synthesized_answer = None
        if return_synthesized:
            print("📝 Синтез финального ответа...")
            synthesized_answer = self.aggregator.aggregate_and_synthesize(
                query,
                all_results
            )

        return {
            "query": query,
            "subqueries": subqueries,
            "results": all_results,
            "synthesized_answer": synthesized_answer
        }

    def answer_question(self, query: str) -> str:
        """Простой интерфейс: вопрос -> ответ"""
        result = self.process_complex_query(query, return_synthesized=True)
        return result["synthesized_answer"]


# =========================
# Функция для красивого вывода
# =========================

def pretty_print_results(result: Dict[str, Any]):
    """Красивый вывод результатов"""

    print("\n" + "=" * 80)
    print(f"📌 ЗАПРОС: {result['query']}")
    print("=" * 80)

    print("\n🔹 ПОДЗАПРОСЫ:")
    for sq in result['subqueries']:
        print(f"  • {sq['subquery']} (приоритет: {sq['priority']})")

    print("\n🔹 НАЙДЕНО ФРАГМЕНТОВ:")
    total = sum(len(results) for results in result['results'].values())
    print(f"  Всего: {total} из {len(result['results'])} подзапросов")

    if result['synthesized_answer']:
        print("\n" + "=" * 80)
        print("📄 СИНТЕЗИРОВАННЫЙ ОТВЕТ:")
        print("=" * 80)
        print(result['synthesized_answer'])

    print("\n" + "=" * 80)


# =========================
# Main с использованием LLM
# =========================

if __name__ == "__main__":
    import os

    # Загружаем индекс и модели
    if os.path.exists(INDEX_PATH) and os.path.exists(DATA_PATH):
        print("📂 Загружаем существующий индекс...")
        index, data, embedding_model, reranker = load_all()
    else:
        print("❌ Индекс не найден. Сначала создайте его.")
        exit(1)

    # Конфигурируем LLM
    # Вариант 1: OpenAI
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",  # или "gpt-3.5-turbo"
        api_key=os.getenv("OPENAI_API_KEY", "your-key-here")
    )

    # Вариант 2: Локальная модель через Ollama
    # llm_config = LLMConfig(
    #     provider="local",
    #     model="qwen2.5:7b",
    #     base_url="http://localhost:11434/v1",
    #     api_key="ollama"
    # )

    # Создаем RAG систему
    rag = LLMEnhancedRAG(
        index=index,
        data=data,
        embedding_model=embedding_model,
        reranker=reranker,
        llm_config=llm_config
    )

    # Тестируем сложные запросы
    test_queries = [
        "капитал за 2024 и 2025, облигации",
        "прибыль и активы в 2025 году по сравнению с 2024",
        "структура кредитного портфеля и риски в 2025",
    ]

    for query in test_queries:
        print("\n" + "🚀" * 40)
        result = rag.process_complex_query(query)
        pretty_print_results(result)

        input("\nНажмите Enter для следующего запроса...")

    # Интерактивный режим
    print("\n" + "💬" * 40)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ (введите 'exit' для выхода)")
    print("💬" * 40)

    while True:
        query = input("\n🤔 Ваш вопрос: ").strip()

        if query.lower() == 'exit':
            break

        result = rag.process_complex_query(query)
        pretty_print_results(result)