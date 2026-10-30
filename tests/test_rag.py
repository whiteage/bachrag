import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cfg import LLMConfig
from rag import LLMClient, QueryDecomposer, ResultAggregator, SimpleLLMEnhancedRAG


class RagTests(unittest.TestCase):
    def test_decomposition_normalizes_and_limits_untrusted_search_counts(self):
        client = Mock()
        client.complete.return_value = json.dumps({"subqueries": [
            None, {"subquery": " "}, {"subquery": "capital", "search_k": 9999},
            {"subquery": "capital"}, {"subquery": "profit", "search_k": "invalid"},
        ]})
        results = QueryDecomposer(client).decompose("original")
        self.assertEqual([item["subquery"] for item in results], ["capital", "profit"])
        self.assertEqual([item["search_k"] for item in results], [10, 3])

    def test_malformed_or_empty_decomposition_falls_back_to_original_question(self):
        for response in ("not JSON", "[]", "null", '{"subqueries": null}', '{"subqueries": []}'):
            with self.subTest(response=response):
                client = Mock()
                client.complete.return_value = response
                results = QueryDecomposer(client).decompose("original")
                self.assertEqual(results[0]["subquery"], "original")

    def test_fenced_json_is_accepted(self):
        client = Mock()
        client.complete.return_value = '```json\n{"subqueries": [{"subquery": "capital"}]}\n```'
        self.assertEqual(QueryDecomposer(client).decompose("original")[0]["subquery"], "capital")

    def test_empty_retrieval_does_not_ask_llm_to_invent_answer(self):
        client = Mock()
        answer = ResultAggregator(client).aggregate_and_synthesize("question", {})
        self.assertIn("не найдено", answer)
        client.complete.assert_not_called()

    def test_complex_query_limits_unique_evidence_and_preserves_response_schema(self):
        client = Mock()
        client.complete.return_value = "answer"
        rag = SimpleLLMEnhancedRAG(None, [], None, None, LLMConfig(), llm_client=client)
        rag.decomposer.decompose = Mock(return_value=[
            {"subquery": "first", "search_k": 10},
            {"subquery": "second", "search_k": 10},
        ])
        chunks = [{"id": i, "text": f"text {i}", "rerank_score": i,
                   "metadata": {"source": "report.pdf", "page": i + 1}} for i in range(12)]
        rag.search_subquery = Mock(side_effect=[chunks[:10], chunks[2:]])
        result = rag.process_complex_query(" question ")
        ids = {chunk["id"] for items in result["results"].values() for chunk in items}
        self.assertEqual(ids, set(range(4, 12)))
        self.assertEqual(result["query"], "question")
        self.assertEqual(result["synthesized_answer"], "answer")
        # The same chunk found for two subqueries is sent to synthesis only once.
        self.assertEqual(client.complete.call_args.args[0].count("text 5"), 1)

    def test_json_mode_is_used_only_for_structured_calls(self):
        api = Mock()
        api.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        )
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=Mock(return_value=api))}):
            client = LLMClient(LLMConfig(api_key="unit-test-placeholder"))
        client.complete("question", use_json=True)
        options = api.chat.completions.create.call_args.kwargs
        self.assertEqual(options["response_format"], {"type": "json_object"})
        self.assertIn("JSON", options["messages"][-1]["content"])
        client.complete("question")
        self.assertNotIn("response_format", api.chat.completions.create.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
