import gc
import os
import hashlib
import pypdf
from transformers import AutoTokenizer
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker

SMALL_DOC_PAGES = 10
BATCH_SIZE = 10
MAX_TOKENS_TEXT = 512       # текст — короткие чанки, точнее retrieval
MAX_TOKENS_TABLE = 1900     # таблицы — как можно целиком
OVERLAP_TOKENS = 50         # перекрытие для текстовых чанков
TOKENIZER_NAME = "ai-sage/Giga-Embeddings-instruct"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


# ── Конвертация ────────────────────────────────────────────────────────────────

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


# ── Чанкинг по типу контента ───────────────────────────────────────────────────

def chunk_document(doc) -> list:
    """
    Таблицы и текст чанкуются с разными лимитами токенов.
    HybridChunker уже знает тип каждого блока — используем это.
    """
    text_chunker  = HybridChunker(max_tokens=MAX_TOKENS_TEXT,  tokenizer=tokenizer)
    table_chunker = HybridChunker(max_tokens=MAX_TOKENS_TABLE, tokenizer=tokenizer)

    text_chunks  = list(text_chunker.chunk(doc))
    table_chunks = list(table_chunker.chunk(doc))
    print("TABLE")
    print(table_chunks)
    # Берём таблицы из table_chunker, текст из text_chunker
    result = []
    for chunk in table_chunks:
        doc_item = chunk.meta.doc_items[0] if chunk.meta.doc_items else None
        if doc_item and "table" in str(doc_item.label).lower():
            result.append(("table", chunk))

    for chunk in text_chunks:
        doc_item = chunk.meta.doc_items[0] if chunk.meta.doc_items else None
        if not doc_item or "table" not in str(doc_item.label).lower():
            result.append(("text", chunk))

    return result


# ── Батчевая обработка с overlap страниц ──────────────────────────────────────

def process_pdf(pdf_path: str) -> list:
    num_pages = get_pdf_pages(pdf_path)
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
    seen = set()
    raw_chunks = []  # список (type_str, chunk)

    print(f"[INFO] {doc_id}: {num_pages} страниц")

    if num_pages <= SMALL_DOC_PAGES:
        doc = convert_pages(pdf_path)
        raw_chunks = chunk_document(doc)
    else:
        for start in range(1, num_pages + 1, BATCH_SIZE):
            # overlap -1 страница чтобы таблицы на границе батчей собирались целиком
            overlap_start = max(1, start - 1)
            end = min(start + BATCH_SIZE - 1, num_pages)
            print(f"[INFO] Батч {overlap_start}-{end}")

            try:
                doc = convert_pages(pdf_path, page_range=(overlap_start, end))
                batch_chunks = chunk_document(doc)
            except MemoryError:
                print(f"[WARN] MemoryError, уменьшаем батч")
                mid = (overlap_start + end) // 2
                doc = convert_pages(pdf_path, page_range=(overlap_start, mid))
                batch_chunks = chunk_document(doc)
                doc = convert_pages(pdf_path, page_range=(mid + 1, end))
                batch_chunks += chunk_document(doc)
            except Exception as e:
                print(f"[ERROR] Батч {overlap_start}-{end}: {e}")
                continue

            # Дедупликация по md5 чтобы убрать дубли из overlap
            for type_str, chunk in batch_chunks:
                key = hashlib.md5(chunk.text.encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    raw_chunks.append((type_str, chunk))

    enriched = enrich_metadata(raw_chunks, doc_id, pdf_path)
    merged   = merge_table_chunks(enriched)
    split    = split_oversized_chunks(merged)
    return reindex(split, doc_id)


# ── Метаданные ─────────────────────────────────────────────────────────────────

def enrich_metadata(raw_chunks: list, doc_id: str, source: str) -> list:
    enriched = []
    for i, (type_str, chunk) in enumerate(raw_chunks):
        doc_item = chunk.meta.doc_items[0] if chunk.meta.doc_items else None
        enriched.append({
            "content": chunk.text,
            "metadata": {
                "doc_id":      doc_id,
                "chunk_id":    f"{doc_id}_chunk_{i:03d}",
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "source":      source,
                "page":        doc_item.prov[0].page_no if doc_item else None,
                "type":        type_str,
                "breadcrumbs": " → ".join(chunk.meta.headings) if chunk.meta.headings else "",
                "section":     chunk.meta.headings[-1] if chunk.meta.headings else "",
                "prev_chunk_id": f"{doc_id}_chunk_{i-1:03d}" if i > 0 else None,
                "next_chunk_id": f"{doc_id}_chunk_{i+1:03d}" if i < len(raw_chunks) - 1 else None,
            }
        })
    return enriched


# ── Склейка табличных чанков ───────────────────────────────────────────────────

def merge_table_chunks(chunks: list) -> list:
    """Склеивает соседние табличные чанки с одной страницы и одной секции."""
    merged = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if c["metadata"]["type"] == "table":
            group = [c]
            j = i + 1
            while (
                j < len(chunks)
                and chunks[j]["metadata"]["type"] == "table"
                and chunks[j]["metadata"]["page"] == c["metadata"]["page"]
                and chunks[j]["metadata"]["section"] == c["metadata"]["section"]
            ):
                group.append(chunks[j])
                j += 1

            if len(group) > 1:
                print(f"[INFO] Склеиваем {len(group)} табличных чанков "
                      f"(стр. {c['metadata']['page']}, «{c['metadata']['section']}»)")

            merged.append({
                "content": "\n".join(g["content"] for g in group),
                "metadata": {**c["metadata"],
                             "next_chunk_id": group[-1]["metadata"]["next_chunk_id"]},
            })
            i = j
        else:
            merged.append(c)
            i += 1
    return merged


# ── Разрезка чанков превышающих лимит ─────────────────────────────────────────

def split_oversized_chunks(chunks: list) -> list:
    """
    Режет чанки превышающие лимит по предложениям.
    Для текста добавляет overlap между частями.
    """
    result = []

    for c in chunks:
        is_table = c["metadata"]["type"] == "table"
        max_tok   = MAX_TOKENS_TABLE if is_table else MAX_TOKENS_TEXT
        tokens    = tokenizer.encode(c["content"])

        if len(tokens) <= max_tok:
            result.append(c)
            continue

        print(f"[INFO] Разрезаем {c['metadata']['chunk_id']}: "
              f"{len(tokens)} токенов → части по {max_tok}")

        sentences = [s.strip() + "." for s in c["content"].split(". ") if s.strip()]
        sub_chunks = []
        current: list[str] = []
        current_tok = 0

        for sentence in sentences:
            stok = len(tokenizer.encode(sentence))
            if current_tok + stok > max_tok and current:
                sub_chunks.append(" ".join(current))

                if is_table:
                    # Таблицы — без overlap, строки самодостаточны
                    current = [sentence]
                    current_tok = stok
                else:
                    # Текст — откатываемся на OVERLAP_TOKENS
                    overlap_sents: list[str] = []
                    overlap_tok = 0
                    for s in reversed(current):
                        st = len(tokenizer.encode(s))
                        if overlap_tok + st > OVERLAP_TOKENS:
                            break
                        overlap_sents.insert(0, s)
                        overlap_tok += st
                    current = overlap_sents + [sentence]
                    current_tok = overlap_tok + stok
            else:
                current.append(sentence)
                current_tok += stok

        if current:
            sub_chunks.append(" ".join(current))

        for part_i, content in enumerate(sub_chunks):
            result.append({
                "content": content,
                "metadata": {
                    **c["metadata"],
                    "chunk_id":    f"{c['metadata']['chunk_id']}_part{part_i}",
                    "split_part":  part_i,
                    "total_parts": len(sub_chunks),
                },
            })

    return result


# ── Финальная перенумерация ────────────────────────────────────────────────────

def reindex(chunks: list, doc_id: str) -> list:
    total = len(chunks)
    for idx, c in enumerate(chunks):
        c["metadata"].update({
            "chunk_index":   idx,
            "total_chunks":  total,
            "chunk_id":      f"{doc_id}_chunk_{idx:03d}",
            "prev_chunk_id": f"{doc_id}_chunk_{idx-1:03d}" if idx > 0 else None,
            "next_chunk_id": f"{doc_id}_chunk_{idx+1:03d}" if idx < total - 1 else None,
        })
    return chunks

# ── Точка входа ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    a = convert_pages("chunker/обобщенный отчет сбер за 2025.pdf", [5, 5])
    chunks = process_pdf("обобщенный отчет сбер за 2025.pdf")
    print(f"\nИтого чанков: {len(chunks)}")

    # Статистика по размерам
    sizes = [len(tokenizer.encode(c["content"])) for c in chunks]
    print(f"Мин: {min(sizes)} | Макс: {max(sizes)} | "
          f"Среднее: {sum(sizes) // len(sizes)} токенов")

    for chunk in chunks:
        print(chunk["metadata"])
        print(chunk["content"])
        print("---")
