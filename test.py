# =========================
# Исправленный LLMClient для DeepSeek
# =========================
import json
from typing import Dict, Any, List

import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from cfg import LLMConfig
from embedder import load_all, DATA_PATH, INDEX_PATH, enhanced_search_with_rerank


class LLMClient:
    """Универсальный клиент для работы с LLM"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=config.base_url
        )

    def complete(self, prompt: str, system_prompt: str = None, use_json: bool = False) -> str:
        """Базовый вызов LLM"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Для JSON ответов добавляем указание в промпт
            if use_json:
                # DeepSeek требует слово "json" в промпте
                if "json" not in prompt.lower():
                    prompt += "\n\nRespond with valid JSON only."
                    messages[-1]["content"] = prompt

                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"}
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )

            return response.choices[0].message.content

        except Exception as e:
            print(f"⚠️ Ошибка при вызове LLM: {e}")
            raise e


# =========================
# Исправленный QueryDecomposer
# =========================

class QueryDecomposer:
    """Разбивает сложный запрос на подзапросы (оптимизировано для DeepSeek)"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def decompose(self, query: str) -> List[Dict[str, Any]]:
        """
        Возвращает список подзапросов с метаданными
        """
        system_prompt = """Ты - система анализа поисковых запросов для финансовых документов. 
Разбивай сложные запросы на простые подзапросы. 
Всегда возвращай строго валидный JSON без дополнительного текста."""

        prompt = f"""Разбей запрос на подзапросы для поиска в финансовом отчете.

ЗАПРОС: {query}

Требования:
1. Для каждой сущности (капитал, облигации, прибыль и т.д.) создай отдельный подзапрос
2. Если указаны года - создай отдельные подзапросы для каждого года
3. Укажи приоритет от 1 до 10 (10 - самый важный)
4. Для финансовых показателей ставь высокий приоритет

Верни ответ в формате JSON. Пример:
{{
    "subqueries": [
        {{
            "subquery": "размер капитала 2024 год",
            "entity_type": "капитал",
            "year": 2024,
            "priority": 10,
            "search_k": 3
        }}
    ]
}}

Верни ТОЛЬКО JSON для запроса: "{query}"
"""

        # Используем JSON режим
        response = self.llm.complete(prompt, system_prompt, use_json=True)

        # Очищаем ответ от маркдауна
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            result = json.loads(response)
            return result.get("subqueries", [])
        except json.JSONDecodeError as e:
            print(f"⚠️ Ошибка парсинга JSON от DeepSeek: {e}")
            print(f"Ответ: {response[:200]}...")
            # Fallback: простое разбиение
            return [{"subquery": query, "entity_type": "general", "year": None, "priority": 10, "search_k": 5}]


# =========================
# Исправленный ResultAggregator
# =========================

class ResultAggregator:
    """Объединяет результаты подзапросов (оптимизировано для DeepSeek)"""

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
        # Подготавливаем контекст
        context_parts = []

        for subquery, results in subqueries_results.items():
            if not results:
                continue

            context_parts.append(f"\n=== Результаты: '{subquery}' ===\n")

            for i, r in enumerate(results[:3], 1):
                meta = r.get("metadata", {})
                text = r.get("text", "")
                score = r.get("rerank_score", 0)

                context_parts.append(
                    f"[Фрагмент {i}, стр.{meta.get('page', 'N/A')}, релевантность: {score:.3f}]\n"
                    f"{text}\n"
                )

        context = "\n".join(context_parts)

        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        # Если нет результатов
        if not context_parts:
            return "К сожалению, по вашему запросу не найдено релевантной информации в документах."

        # Промпт для DeepSeek (без JSON формата, так как нам нужен обычный текст)
        system_prompt = """Ты - финансовый аналитик. Отвечай строго на основе предоставленных фрагментов.
Если данных недостаточно - укажи это. Выделяй конкретные числа и даты.
Форматируй ответ красиво, используй маркдаун для таблиц и списков."""

        prompt = f"""На основе фрагментов из финансового отчета ответь на вопрос.

ВОПРОС: {original_query}

ФРАГМЕНТЫ:
{context}

Требования к ответу:
1. Если есть данные за разные годы - сравни их в виде таблицы
2. Указывай конкретные числа с единицами измерения (млрд руб, %)
3. Указывай номера страниц источников в скобках
4. Если какой-то информации нет - отметь это
5. Структурируй ответ: сначала основные цифры, потом детали

Ответ:"""

        # Не используем JSON формат для финального ответа
        response = self.llm.complete(prompt, system_prompt, use_json=False)
        return response


# =========================
# Исправленный ChunkSelector
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
        for i, chunk in enumerate(candidates[:15]):  # Оцениваем только первые 15
            text = chunk.get("text", "")[:300]  # Превью для LLM
            chunks_text.append(f"[{i}] {text}")

        system_prompt = """Ты - система оценки релевантности. 
Оцени какие фрагменты наиболее полно отвечают на запрос.
Верни JSON с массивом индексов лучших фрагментов."""

        prompt = f"""Запрос: "{query}"

Фрагменты для оценки:
{chr(10).join(chunks_text)}

Выбери {max_chunks} наиболее релевантных фрагментов. 
Верни ответ в формате JSON:
{{"selected_indices": [0, 3, 5]}}

Верни ТОЛЬКО JSON с индексами выбранных фрагментов."""

        response = self.llm.complete(prompt, system_prompt, use_json=True)

        # Очищаем ответ
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        try:
            data = json.loads(response)
            selected = [candidates[i] for i in data["selected_indices"] if i < len(candidates)]
            return selected[:max_chunks]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"⚠️ Ошибка при выборе чанков: {e}")
            # Fallback: берем топ по скорам
            candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            return candidates[:max_chunks]


# =========================
# Упрощенная версия для быстрого старта (если не нужен сложный выбор)
# =========================

class SimpleLLMEnhancedRAG:
    """Упрощенная версия без сложного выбора чанков"""

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

    def search_subquery(self, subquery: str, k: int = 5) -> List[Dict]:
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

    def process_complex_query(self, query: str) -> Dict[str, Any]:
        """
        Полный цикл обработки сложного запроса (упрощенный)
        """
        print(f"🤔 Обработка запроса: {query}")

        # Шаг 1: Декомпозиция через LLM
        print("📋 Декомпозиция запроса...")
        subqueries = self.decomposer.decompose(query)

        if not subqueries:
            subqueries = [{"subquery": query, "search_k": 5}]

        # Шаг 2: Поиск по всем подзапросам
        print(f"🔍 Поиск по {len(subqueries)} подзапросам...")
        all_results = {}
        all_chunks = []
        seen_ids = set()

        for sq in subqueries:
            subquery_text = sq["subquery"]
            k = sq.get("search_k", 3)

            print(f"  - {subquery_text} (k={k})")
            results = self.search_subquery(subquery_text, k=k)

            if results:
                all_results[subquery_text] = results
                for chunk in results:
                    if chunk["id"] not in seen_ids:
                        seen_ids.add(chunk["id"])
                        all_chunks.append(chunk)

        # Шаг 3: Сортируем все уникальные чанки по реранк скору
        all_chunks.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        # Перегруппируем топ-8 чанков
        top_chunks = all_chunks[:8]
        filtered_results = {}

        for sq, results in all_results.items():
            filtered = [r for r in results if r in top_chunks]
            if filtered:
                filtered_results[sq] = filtered

        # Шаг 4: Синтез ответа
        print("📝 Синтез финального ответа...")
        synthesized_answer = self.aggregator.aggregate_and_synthesize(
            query,
            filtered_results or all_results  # если filtered пустой, берем все
        )

        return {
            "query": query,
            "subqueries": subqueries,
            "results": filtered_results or all_results,
            "synthesized_answer": synthesized_answer
        }


# =========================
# Main с исправлениями
# =========================

if __name__ == "__main__":
    from dotenv import load_dotenv

    # Загружаем переменные окружения
    load_dotenv()

    # Загружаем индекс и модели
    if os.path.exists(INDEX_PATH) and os.path.exists(DATA_PATH):
        print("📂 Загружаем существующий индекс...")
        index, data, embedding_model, reranker = load_all()
    else:
        print("❌ Индекс не найден. Сначала создайте его.")
        exit(1)

    # Конфигурация DeepSeek
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        api_key = input("Введите DeepSeek API ключ: ").strip()

    llm_config = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
        api_key=api_key,
        temperature=0.0,
        max_tokens=2000
    )

    # Создаем RAG систему (используем упрощенную версию)
    rag = SimpleLLMEnhancedRAG(
        index=index,
        data=data,
        embedding_model=embedding_model,
        reranker=reranker,
        llm_config=llm_config
    )

    # Тестируем
    query = "на сколько процентов увеличился капитал с 2024 по 2025"

    print("\n" + "🚀" * 40)
    print(f"ТЕСТОВЫЙ ЗАПРОС: {query}")
    print("🚀" * 40)

    result = rag.process_complex_query(query)

    # Выводим результат
    if result['synthesized_answer']:
        print("\n" + "=" * 80)
        print("📄 ФИНАЛЬНЫЙ ОТВЕТ:")
        print("=" * 80)
        print(result['synthesized_answer'])

    print("\n✅ Готово!")