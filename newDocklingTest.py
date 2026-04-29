"""
chunker.py — универсальный чанкер для PDF через docling.

Поддерживает:
- Финансовые отчёты (таблицы с числовыми данными)
- Любые другие документы (текстовые таблицы → markdown)
- Автодетект структуры заголовка таблицы
- Контекст секции документа в каждом чанке

Исправления по сравнению с предыдущей версией:
- chunker.chunk(doc) вызывается один раз (материализация в список)
- id(item) заменён на стабильный self_ref для table_contexts
- merge_broken_rows корректно объединяет значения (первые непустые)
- _get_chunk_id использует hashlib.md5 вместо hash()
- Короткие текстовые чанки (< MIN_TEXT_LENGTH символов) отфильтровываются
- detect_value_columns не помечает пустые колонки как служебные
- Текстовые чанки получают overlap через _apply_text_overlap()
- pbar.update считает только новые страницы батча
"""

import gc
import hashlib
import json
import os
import re
import warnings
from typing import Optional

import pypdf
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HybridChunker
from docling_core.types.doc import DocItemLabel
from langchain_core.documents import Document
from tqdm import tqdm

from dockilingTest import tokenizer, MAX_TOKENS_TEXT, SMALL_DOC_PAGES, BATCH_SIZE

# Минимальная длина текстового чанка в символах
MIN_TEXT_LENGTH = 80
# Количество символов overlap между соседними текстовыми чанками
TEXT_OVERLAP_CHARS = 150


# ══════════════════════════════════════════════════════════════════════════════
# КОНВЕРТАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def get_pdf_pages(pdf_path: str) -> int:
    return len(pypdf.PdfReader(pdf_path).pages)


def make_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def convert_pages(pdf_path: str, page_range=None):
    kwargs = {"source": pdf_path}
    if page_range:
        kwargs["page_range"] = page_range
    result = make_converter().convert(**kwargs)
    doc = result.document
    del result
    gc.collect()
    return doc


# ══════════════════════════════════════════════════════════════════════════════
# ПАЙПЛАЙН
# ══════════════════════════════════════════════════════════════════════════════

def _get_chunk_id(chunk: Document) -> str:
    """Детерминированный ID чанка через MD5 содержимого."""
    return hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()


def process_pdf(pdf_path: str) -> list[Document]:
    num_pages = get_pdf_pages(pdf_path)
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"[INFO] {doc_id}: {num_pages} страниц")

    all_chunks: list[Document] = []
    seen_ids: set[str] = set()

    def _add_chunks(chunks: list[Document]):
        for chunk in chunks:
            cid = _get_chunk_id(chunk)
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(chunk)

    if num_pages <= SMALL_DOC_PAGES:
        doc = convert_pages(pdf_path)
        _add_chunks(chunk_page(doc, source=pdf_path))
        return all_chunks

    num_batches = (num_pages + BATCH_SIZE - 1) // BATCH_SIZE
    with tqdm(total=num_pages, desc=f"Обработка {doc_id}", unit="стр") as pbar:
        for batch_idx, start in enumerate(range(1, num_pages + 1, BATCH_SIZE)):
            overlap_start = max(1, start - 1)
            end = min(start + BATCH_SIZE - 1, num_pages)
            tqdm.write(
                f"[INFO] Батч {batch_idx + 1}/{num_batches}: стр. {overlap_start}-{end}"
            )
            try:
                doc = convert_pages(pdf_path, page_range=(overlap_start, end))
                _add_chunks(chunk_page(doc, source=pdf_path))
            except MemoryError:
                tqdm.write(f"[WARN] MemoryError, дробим батч {overlap_start}-{end}...")
                _process_batch_recursive(pdf_path, overlap_start, end, _add_chunks)
            except Exception as e:
                tqdm.write(f"[ERROR] Батч {overlap_start}-{end}: {e}")
            finally:
                # Обновляем только новые страницы батча, не считая overlap-страницу
                pbar.update(end - start + 1)

    return all_chunks


def _process_batch_recursive(pdf_path, start, end, add_fn):
    if start >= end:
        return
    mid = (start + end) // 2
    for s, e in [(start, mid), (mid + 1, end)]:
        try:
            doc = convert_pages(pdf_path, page_range=(s, e))
            add_fn(chunk_page(doc, source=pdf_path))
        except MemoryError:
            _process_batch_recursive(pdf_path, s, e, add_fn)
        except Exception as ex:
            warnings.warn(f"[ERROR] Батч {s}-{e}: {ex}")


# ══════════════════════════════════════════════════════════════════════════════
# ПОСТРОЕНИЕ ГРИДА
# ══════════════════════════════════════════════════════════════════════════════

def build_grid(table_cells):
    max_row = max(c.end_row_offset_idx for c in table_cells)
    max_col = max(c.end_col_offset_idx for c in table_cells)
    grid = [[None] * max_col for _ in range(max_row)]
    for cell in table_cells:
        for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
            for c in range(cell.start_col_offset_idx, cell.end_col_offset_idx):
                grid[r][c] = cell
    return grid


# ══════════════════════════════════════════════════════════════════════════════
# АВТОДЕТЕКТ СТРУКТУРЫ
# ══════════════════════════════════════════════════════════════════════════════

def detect_header_rows(grid, max_check: int = 4) -> int:
    """
    Определяет количество строк заголовка по наличию colspan.
    Строки с colspan → часть заголовка.
    Первая строка без colspan после строк с colspan → последняя строка заголовка.
    """
    had_span = False
    for r in range(min(max_check, len(grid))):
        seen_ids: set[int] = set()
        has_colspan = False
        for cell in grid[r]:
            if cell is None:
                continue
            cid = id(cell)
            if cid in seen_ids:
                has_colspan = True
                break
            seen_ids.add(cid)

        if had_span and not has_colspan:
            return r + 1
        if has_colspan:
            had_span = True

    return 1


def find_name_col_count(grid, header_rows: int) -> int:
    """Сколько левых колонок занимает ячейка с названием строки."""
    if header_rows >= len(grid):
        return 1
    first_cell = grid[header_rows][0]
    if first_cell is None:
        return 1
    return first_cell.end_col_offset_idx - first_cell.start_col_offset_idx


def is_numeric_table(rows: list[dict]) -> bool:
    """
    Определяет числовая ли таблица.
    Если хотя бы 30% значений — числа, считаем числовой.
    """
    total, numeric = 0, 0
    for row in rows:
        for v in row["values"]:
            if v:
                total += 1
                parsed = parse_number(v)
                if isinstance(parsed, float):
                    numeric += 1
    if total == 0:
        return False
    return (numeric / total) >= 0.3


# ══════════════════════════════════════════════════════════════════════════════
# ИЗВЛЕЧЕНИЕ ЗАГОЛОВКА И СТРОК
# ══════════════════════════════════════════════════════════════════════════════

def extract_header(grid, header_rows: int = 1) -> list[str | None]:
    """
    Собирает заголовок из нескольких строк.
    Дедупликация по id ячейки — корректно обрабатывает colspan/rowspan.
    """
    n_cols = len(grid[0])
    columns = []
    for col_idx in range(n_cols):
        parts = []
        seen_cell_ids: set[int] = set()
        for r in range(min(header_rows, len(grid))):
            cell = grid[r][col_idx]
            if cell is None:
                continue
            cid = id(cell)
            if cid in seen_cell_ids:
                continue
            seen_cell_ids.add(cid)
            if cell.text and cell.text.strip():
                parts.append(cell.text.strip())
        columns.append(" ".join(parts) if parts else None)
    return columns


def extract_rows(grid, header_rows: int = 1,
                 name_cols: int = 1) -> list[dict]:
    """
    Извлекает строки данных.
    name_cols — сколько левых колонок занимает название строки.
    """
    rows = []
    current_section = None
    buffer_row = None

    for r in range(header_rows, len(grid)):
        row = grid[r]
        first = row[0]

        if not first:
            continue

        if first.row_section:
            current_section = first.text.strip()
            continue

        name = first.text.strip()
        values = [
            cell.text.strip() if cell and cell.text else None
            for cell in row[name_cols:]
        ]
        has_data = any(v for v in values)

        if not has_data:
            if buffer_row:
                buffer_row["name"] += " " + name
            continue

        buffer_row = {
            "section": current_section,
            "name": name,
            "values": values,
        }
        rows.append(buffer_row)

    return rows


def merge_broken_rows(rows: list[dict]) -> list[dict]:
    """
    Склеивает строки, название которых разбито на несколько физических строк.
    Значения берутся из первой строки, дополняются первыми непустыми из продолжения.
    """
    merged, buffer = [], None
    for row in rows:
        if buffer is None:
            buffer = dict(row)
            buffer["values"] = list(row["values"])
            continue
        if row["name"] and row["name"][0].islower():
            buffer["name"] += " " + row["name"]
            # Берём значения из продолжения только там, где в буфере пусто
            for i, v in enumerate(row["values"]):
                if i < len(buffer["values"]) and not buffer["values"][i] and v:
                    buffer["values"][i] = v
        else:
            merged.append(buffer)
            buffer = dict(row)
            buffer["values"] = list(row["values"])
    if buffer:
        merged.append(buffer)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def parse_number(x):
    if not x:
        return None
    x = x.replace("\u00a0", "").replace(" ", "").replace(",", ".")
    if "(" in x:
        x = "-" + x.replace("(", "").replace(")", "")
    try:
        return float(x)
    except Exception:
        return x


def _parse_year(col_name: str | None) -> int | None:
    if not col_name:
        return None
    m = re.search(r"\b(20\d{2})\b", col_name)
    return int(m.group(1)) if m else None


def _clean_col_name(col: str | None) -> str | None:
    """Сокращает длинное название колонки до года если возможно."""
    year = _parse_year(col)
    return str(year) if year else col


def detect_value_columns(rows: list[dict], columns: list[str | None],
                         max_ref_value: float = 999.0) -> set[str | None]:
    """
    Служебные колонки (номера примечаний) — только целые числа <= max_ref_value.
    Пустые и текстовые колонки НЕ считаются служебными — они могут нести смысл.
    """
    service_cols: set[str | None] = set()
    for col_idx, col_name in enumerate(columns):
        numeric_values = []
        has_any_value = False
        for row in rows:
            raw = row["values"][col_idx] if col_idx < len(row["values"]) else None
            if raw:
                has_any_value = True
            parsed = parse_number(raw)
            if isinstance(parsed, float):
                numeric_values.append(parsed)

        # Пустая колонка — пропускаем, не помечаем как служебную
        if not has_any_value:
            continue

        # Колонка с нечисловыми значениями — не служебная
        if not numeric_values:
            continue

        # Служебная: все значения целые и <= max_ref_value
        if (all(v == int(v) for v in numeric_values) and
                all(abs(v) <= max_ref_value for v in numeric_values)):
            service_cols.add(col_name)

    return service_cols


def _dedup_ordered(lst: list) -> list:
    """Убирает дубли сохраняя порядок."""
    seen: set = set()
    return [x for x in lst if not (x in seen or seen.add(x))]  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# ФОРМИРОВАНИЕ ЧАНКОВ ДЛЯ ЧИСЛОВЫХ ТАБЛИЦ
# ══════════════════════════════════════════════════════════════════════════════

def make_row_chunks(rows, columns, value_columns, table_name,
                    unit=None, doc_context: str = "") -> list[dict]:
    result = []
    col_index = {col: i for i, col in enumerate(columns)}
    for row in rows:
        values_dict = {
            col: parse_number(row["values"][col_index[col]])
            for col in value_columns
        }
        values_text = ", ".join(
            f"{_clean_col_name(col)} — {values_dict[col]}"
            f"{(' ' + unit) if unit else ''}"
            for col in value_columns
        )
        parts = []
        if doc_context:
            parts.append(f"[{doc_context}]")
        section = row["section"] or ""
        if section:
            parts.append(section)
        parts.append(f"{row['name']}: {values_text}")

        text = ". ".join(parts)
        result.append({
            "type": "table_row", "table": table_name,
            "section": row["section"], "row": row["name"],
            "values": values_dict, "text": text,
            "doc_context": doc_context,
        })
    return result


def make_cell_chunks(rows, columns, value_columns, table_name,
                     unit=None, doc_context: str = "") -> list[dict]:
    result = []
    col_index = {col: i for i, col in enumerate(columns)}
    for row in rows:
        for col in value_columns:
            value = parse_number(row["values"][col_index[col]])
            col_label = _clean_col_name(col)

            parts = []
            if doc_context:
                parts.append(f"[{doc_context}]")
            section = row["section"] or ""
            if section:
                parts.append(section)
            parts.append(f"{row['name']} на {col_label}: {value}")
            if unit:
                parts[-1] += f" {unit}"

            text = ". ".join(parts)
            result.append({
                "type": "table_cell", "table": table_name,
                "section": row["section"], "row": row["name"],
                "column": col, "value": value,
                "year": _parse_year(col),
                "text": text,
                "doc_context": doc_context,
            })
    return result


def to_documents(chunks, source=None, page=None,
                 unit=None) -> list[Document]:
    return [
        Document(
            page_content=ch["text"],
            metadata={
                "type":        ch["type"],
                "table":       ch["table"],
                "section":     ch["section"],
                "row":         ch.get("row"),
                "column":      ch.get("column"),
                "value":       ch.get("value"),
                "year":        ch.get("year"),
                "source":      source,
                "page":        page,
                "unit":        unit,
                "doc_context": ch.get("doc_context", ""),
            },
        )
        for ch in chunks
    ]


# ══════════════════════════════════════════════════════════════════════════════
# ОБРАБОТКА ТАБЛИЦЫ
# ══════════════════════════════════════════════════════════════════════════════

def process_table(
        table_item,
        table_name: str,
        source=None,
        page=None,
        unit=None,
        max_ref_value: float = 999.0,
        header_rows: Optional[int] = None,
        name_cols: Optional[int] = None,
        doc_context: str = "",
) -> list[Document]:
    """
    Обрабатывает таблицу.

    Числовые таблицы → table_row + table_cell чанки с контекстом.
    Текстовые таблицы → один markdown чанк (универсально для любых документов).
    """
    if hasattr(table_item, "data"):
        table_cells = table_item.data.table_cells
    else:
        raise ValueError(f"Неизвестный тип table_item: {type(table_item)}")

    grid = build_grid(table_cells)

    if header_rows is None:
        header_rows = detect_header_rows(grid)
    if name_cols is None:
        name_cols = find_name_col_count(grid, header_rows)

    header  = extract_header(grid, header_rows=header_rows)
    columns = header[name_cols:]

    rows = extract_rows(grid, header_rows=header_rows, name_cols=name_cols)
    rows = merge_broken_rows(rows)

    if not rows:
        return []

    if not is_numeric_table(rows):
        return _table_to_markdown_doc(
            table_item=table_item,
            table_name=table_name,
            source=source,
            page=page,
            doc_context=doc_context,
        )

    service_cols  = detect_value_columns(rows, columns, max_ref_value=max_ref_value)
    value_columns = _dedup_ordered([c for c in columns if c not in service_cols])

    row_chunks  = make_row_chunks(
        rows, columns, value_columns, table_name, unit, doc_context
    )
    cell_chunks = make_cell_chunks(
        rows, columns, value_columns, table_name, unit, doc_context
    )

    return to_documents(row_chunks + cell_chunks, source=source, page=page, unit=unit)


def _table_to_markdown_doc(
        table_item,
        table_name: str,
        source=None,
        page=None,
        doc_context: str = "",
) -> list[Document]:
    """Конвертирует текстовую таблицу в markdown чанк."""
    try:
        md = table_item.export_to_markdown()
    except Exception:
        return []

    if not md or not md.strip():
        return []

    content = f"[{doc_context}]\n\n{md}" if doc_context else md

    return [Document(
        page_content=content,
        metadata={
            "type":        "table_markdown",
            "table":       table_name,
            "source":      source,
            "page":        page,
            "doc_context": doc_context,
        }
    )]


# ══════════════════════════════════════════════════════════════════════════════
# ОБРАБОТКА ТЕКСТА
# ══════════════════════════════════════════════════════════════════════════════

_TEXT_CHUNKER = None


def _get_text_chunker() -> HybridChunker:
    global _TEXT_CHUNKER
    if _TEXT_CHUNKER is None:
        _TEXT_CHUNKER = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=MAX_TOKENS_TEXT,
        )
    return _TEXT_CHUNKER


def _is_toc_chunk(text: str) -> bool:
    """Определяет является ли чанк оглавлением."""
    dot_ratio = text.count(".") / max(len(text), 1)
    return "......" in text or dot_ratio > 0.25


def _get_doc_context(chunk) -> str:
    """
    Извлекает контекст секции документа из метаданных чанка HybridChunker.
    Возвращает путь заголовков: 'Примечание 6 > Средства в банках'
    """
    headings = getattr(chunk.meta, "headings", None) or []
    return " > ".join(h for h in headings if h and h.strip())


def _get_stable_table_ref(item) -> str | None:
    """
    Возвращает стабильный идентификатор таблицы из документа.
    Использует self_ref если доступен, иначе prov-координаты.
    """
    if hasattr(item, "self_ref") and item.self_ref:
        return str(item.self_ref)
    if hasattr(item, "prov") and item.prov:
        p = item.prov[0]
        return f"p{p.page_no}_b{getattr(p, 'bbox', '')}"
    return None


def _chunk_text_item(chunk, source=None, page=None) -> Document:
    pages = set()
    for item in (chunk.meta.doc_items or []):
        if hasattr(item, "prov") and item.prov:
            for p in item.prov:
                pages.add(p.page_no)

    doc_context = _get_doc_context(chunk)

    return Document(
        page_content=chunk.text,
        metadata={
            "type":        "text",
            "source":      source,
            "page":        page or (min(pages) if pages else None),
            "pages":       sorted(pages) or None,
            "doc_context": doc_context,
        },
    )


def _apply_text_overlap(documents: list[Document],
                        overlap_chars: int = TEXT_OVERLAP_CHARS) -> list[Document]:
    """
    Добавляет overlap между соседними текстовыми чанками.
    Конец предыдущего чанка приклеивается в начало следующего.
    Работает только для чанков с type=text.
    """
    if overlap_chars <= 0:
        return documents

    text_indices = [i for i, d in enumerate(documents) if d.metadata.get("type") == "text"]

    for pos in range(1, len(text_indices)):
        prev_idx = text_indices[pos - 1]
        curr_idx = text_indices[pos]
        prev_text = documents[prev_idx].page_content
        tail = prev_text[-overlap_chars:].strip()
        if tail:
            documents[curr_idx] = Document(
                page_content=tail + " " + documents[curr_idx].page_content,
                metadata=documents[curr_idx].metadata,
            )

    return documents


# ══════════════════════════════════════════════════════════════════════════════
# ЕДИНАЯ ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════════════

def chunk_page(
        doc,
        source: str | None = None,
        page: int | None = None,
        table_name: str = "table",
        unit: str | None = None,
        max_ref_value: float = 999.0,
) -> list[Document]:
    """
    Принимает DoclingDocument одной или нескольких страниц.

    Текст → HybridChunker → Document (type=text)
    Числовые таблицы → table_row + table_cell чанки
    Текстовые таблицы → markdown чанк (type=table_markdown)

    Контекст секции документа (заголовки) добавляется в каждый чанк
    через поле doc_context и в начало page_content таблиц.
    """
    documents: list[Document] = []
    chunker = _get_text_chunker()

    # Материализуем один раз — chunk() не кешируется
    all_chunks = list(chunker.chunk(doc))

    # Собираем контекст таблиц через стабильный self_ref
    table_contexts: dict[str, str] = {}
    for chunk in all_chunks:
        for item in (chunk.meta.doc_items or []):
            if item.label == DocItemLabel.TABLE:
                ref = _get_stable_table_ref(item)
                if ref:
                    ctx = _get_doc_context(chunk)
                    if ctx:
                        table_contexts[ref] = ctx

    # Текстовые чанки
    for chunk in all_chunks:
        labels = {item.label for item in (chunk.meta.doc_items or [])}
        if labels and any(lbl == DocItemLabel.TABLE for lbl in labels):
            continue
        if not chunk.text or not chunk.text.strip():
            continue
        if _is_toc_chunk(chunk.text):
            continue
        if len(chunk.text.strip()) < MIN_TEXT_LENGTH:
            continue
        documents.append(_chunk_text_item(chunk, source=source, page=page))

    # Применяем overlap к текстовым чанкам
    documents = _apply_text_overlap(documents)

    # Таблицы
    for i, table_item in enumerate(doc.tables or []):
        tbl_page = page
        if hasattr(table_item, "prov") and table_item.prov:
            tbl_page = table_item.prov[0].page_no

        tbl_name = (
            f"{table_name}_{i + 1}" if len(doc.tables) > 1 else table_name
        )

        ref = _get_stable_table_ref(table_item)
        doc_context = table_contexts.get(ref, "") if ref else ""

        try:
            documents.extend(process_table(
                table_item=table_item,
                table_name=tbl_name,
                source=source,
                page=tbl_page,
                unit=unit,
                max_ref_value=max_ref_value,
                doc_context=doc_context,
            ))
        except Exception as e:
            warnings.warn(
                f"Не удалось обработать таблицу {tbl_name} стр.{tbl_page}: {e}"
            )

    return documents


# ══════════════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    docs = process_pdf("обобщенный отчет сбер за 2025.pdf")

    print(f"\nВсего чанков: {len(docs)}")

    types: dict[str, int] = {}
    for d in docs:
        t = d.metadata.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"Типы: {types}")

    output_path = "сбер.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata}
             for d in docs],
            f, ensure_ascii=False, indent=2,
        )
    print(f"✅ Сохранено {len(docs)} чанков → {output_path}")