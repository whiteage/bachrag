"""Regression checks for table values and PDF batch failure handling."""

import unittest
from enum import Enum
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import document_processing as processor


def make_table(rows):
    cells = []
    for row_index, row in enumerate(rows):
        for column_index, text in enumerate(row):
            cells.append(SimpleNamespace(
                start_row_offset_idx=row_index,
                end_row_offset_idx=row_index + 1,
                start_col_offset_idx=column_index,
                end_col_offset_idx=column_index + 1,
                row_section=False,
                text=text,
            ))
    return SimpleNamespace(
        data=SimpleNamespace(table_cells=cells),
        export_to_markdown=lambda: "| Name | Value |\n| --- | --- |\n| Alpha | Beta |",
    )


class TableProcessingTests(unittest.TestCase):
    def test_small_year_values_are_retained(self):
        table = make_table([
            ["Показатель", "Примечание", "2024", "2025"],
            ["Прибыль", "4", "12", "23"],
        ])
        documents = processor.process_table(table, "Отчёт", source="report.pdf", page=7)
        cells = [document for document in documents if document.metadata["type"] == "table_cell"]
        self.assertEqual([document.metadata["value"] for document in cells], [12.0, 23.0])
        self.assertEqual([document.metadata["year"] for document in cells], [2024, 2025])
        self.assertTrue(all(document.metadata["page"] == 7 for document in cells))

    def test_duplicate_headers_preserve_both_columns(self):
        table = make_table([["Показатель", "2025", "2025"], ["Прибыль", "12", "23"]])
        documents = processor.process_table(table, "Отчёт")
        cells = [document for document in documents if document.metadata["type"] == "table_cell"]
        self.assertEqual([document.metadata["value"] for document in cells], [12.0, 23.0])
        self.assertNotEqual(cells[0].metadata["column"], cells[1].metadata["column"])

    def test_empty_table_returns_no_chunks(self):
        self.assertEqual(processor.process_table(make_table([]), "empty"), [])

    def test_text_table_uses_markdown(self):
        table = make_table([["Name", "Description"], ["Alpha", "Beta"]])
        documents = processor.process_table(table, "words", doc_context="Section")
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["type"], "table_markdown")
        self.assertIn("[Section]", documents[0].page_content)

    def test_lowercase_rows_with_values_are_not_merged(self):
        rows = [
            {"section": None, "name": "денежные средства", "values": ["12"]},
            {"section": None, "name": "основные средства", "values": ["23"]},
        ]
        self.assertEqual(len(processor.merge_broken_rows(rows)), 2)

    def test_parse_financial_number(self):
        self.assertEqual(processor.parse_number("(1\u202f234,50)"), -1234.5)
        self.assertEqual(processor.parse_number(0), 0.0)
        self.assertIsNone(processor.parse_number(""))


class TextProcessingTests(unittest.TestCase):
    def test_chunker_keeps_text_separate_from_custom_table_processing(self):
        chunking = ModuleType("docling_core.transforms.chunker")
        transformers = ModuleType("transformers")
        chunking.HybridChunker = Mock()
        transformers.AutoTokenizer = Mock()
        tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
        processor._get_text_chunker.cache_clear()
        try:
            with patch.dict("sys.modules", {
                "docling_core.transforms.chunker": chunking,
                "transformers": transformers,
            }):
                processor._get_text_chunker()
            chunking.HybridChunker.assert_called_once_with(
                tokenizer=tokenizer, max_tokens=512, merge_peers=False
            )
        finally:
            processor._get_text_chunker.cache_clear()

    def test_overlap_uses_original_previous_text(self):
        documents = [
            processor.Document(text, {"type": "text", "source": "a.pdf"})
            for text in ("first", "second", "third")
        ]
        result = processor._apply_text_overlap(documents, overlap_chars=100)
        self.assertEqual(result[1].page_content, "first second")
        self.assertEqual(result[2].page_content, "second third")

    def test_overlap_does_not_cross_sections(self):
        documents = [
            processor.Document("first", {"type": "text", "doc_context": "Section A"}),
            processor.Document("second", {"type": "text", "doc_context": "Section B"}),
        ]
        self.assertEqual(processor._apply_text_overlap(documents)[1].page_content, "second")

    def test_short_document_is_not_discarded(self):
        chunk = SimpleNamespace(text="Short but useful", meta=SimpleNamespace(doc_items=[], headings=[]))
        chunker = SimpleNamespace(chunk=lambda document: iter([chunk]))
        with patch.object(processor, "_get_text_chunker", return_value=chunker):
            documents = processor.chunk_page(SimpleNamespace(tables=[]), source="short.pdf")
        self.assertEqual([document.page_content for document in documents], ["Short but useful"])

    def test_equal_content_on_different_pages_keeps_citations(self):
        first = processor.Document("same", {"page": 1, "source": "a.pdf"})
        duplicate = processor.Document("same", {"page": 1, "source": "a.pdf"})
        second = processor.Document("same", {"page": 2, "source": "a.pdf"})
        self.assertEqual(processor._get_chunk_id(first), processor._get_chunk_id(duplicate))
        self.assertNotEqual(processor._get_chunk_id(first), processor._get_chunk_id(second))


class BatchProcessingTests(unittest.TestCase):
    def test_partial_or_failed_conversion_is_not_indexed(self):
        class ConversionStatus(str, Enum):
            SUCCESS = "success"
            PARTIAL_SUCCESS = "partial_success"
            FAILURE = "failure"
            SKIPPED = "skipped"

        module = ModuleType("docling.datamodel.base_models")
        module.ConversionStatus = ConversionStatus
        with patch.dict("sys.modules", {"docling.datamodel.base_models": module}):
            for status in (
                ConversionStatus.PARTIAL_SUCCESS,
                ConversionStatus.FAILURE,
                ConversionStatus.SKIPPED,
            ):
                with self.subTest(status=status):
                    converter = Mock()
                    converter.convert.return_value = SimpleNamespace(status=status, document=object())
                    with patch.object(processor, "make_converter", return_value=converter):
                        with self.assertRaisesRegex(RuntimeError, "did not complete"):
                            processor.convert_pages("report.pdf", page_range=(9, 20))
            document = SimpleNamespace(pages={9: object(), 20: object()})
            converter = Mock()
            converter.convert.return_value = SimpleNamespace(
                status=ConversionStatus.SUCCESS, document=document
            )
            with patch.object(processor, "make_converter", return_value=converter):
                self.assertIs(processor.convert_pages("report.pdf", page_range=(9, 20)), document)
            converter.convert.assert_called_once_with(source="report.pdf", page_range=(9, 20))

    def test_single_page_memory_failure_is_reported(self):
        with patch.object(processor, "convert_pages", side_effect=MemoryError("page too large")):
            with self.assertRaises(MemoryError):
                processor._process_batch_recursive("report.pdf", 2, 2, lambda chunks: None)

    def test_recursive_batch_retries_single_pages(self):
        chunks = []

        def convert(path, page_range):
            if page_range[0] != page_range[1]:
                raise MemoryError("batch too large")
            return page_range[0]

        with patch.object(processor, "convert_pages", side_effect=convert), patch.object(
            processor, "chunk_page", side_effect=lambda page, source: [page]
        ):
            processor._process_batch_recursive("report.pdf", 1, 4, chunks.extend)
        self.assertEqual(chunks, [1, 2, 3, 4])

    def test_conversion_failure_propagates(self):
        with patch.object(processor, "convert_pages", side_effect=ValueError("bad page")):
            with self.assertRaisesRegex(ValueError, "bad page"):
                processor._process_batch_recursive("report.pdf", 1, 4, lambda chunks: None)


if __name__ == "__main__":
    unittest.main()
