import os
import re
from typing import List, Optional

import pypdf
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling_core.transforms.chunker import HybridChunker
from langchain_core.documents import Document

# =========================
# CONFIG
# =========================
MAX_TOKENS = 512
BATCH_SIZE = 10
DEFAULT_UNIT = "млрд руб"  # можно менять

from dockilingTest import tokenizer

# =========================
# CONVERTER
# =========================

def make_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def convert_pdf_batched(path: str, batch_size: int = BATCH_SIZE):
    total_pages = len(pypdf.PdfReader(path).pages)
    converter = make_converter()

    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)

        print(f"[INFO] Pages {start}-{end}")

        result = converter.convert(
            source=path,
            page_range=(start, end)
        )

        yield result.document
        del result

# =========================
# TOKEN SPLITTER
# =========================

def split_by_tokens(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    tokens = tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        last_dot = chunk_text.rfind('.')
        if last_dot > 100:
            chunk_text = chunk_text[:last_dot + 1]
            actual_tokens = tokenizer.encode(chunk_text)
            end = start + len(actual_tokens)

        chunks.append(chunk_text.strip())
        start = end

    return chunks

# =========================
# UTILS
# =========================

def extract_year(text: str):
    m = re.search(r"(20\\d{2})", text)
    return int(m.group(1)) if m else None


def parse_number(x: Optional[str]):
    if not x:
        return None
    x = x.replace("\u00a0", "").replace(" ", "").replace(",", ".")
    if "(" in x:
        x = "-" + x.replace("(", "").replace(")", "")
    try:
        return float(x)
    except Exception:
        return x

# =========================
# TEXT CHUNKING
# =========================

def chunk_text(doc, source: str) -> List[Document]:
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=1024)
    docs = []

    for chunk in chunker.chunk(doc):
        if not chunk.text or not chunk.text.strip():
            continue

        sub_chunks = split_by_tokens(chunk.text)

        for sub in sub_chunks:
            docs.append(Document(
                page_content=sub,
                metadata={
                    "type": "text",
                    "source": source,
                    "headings": chunk.meta.headings or []
                }
            ))

    return docs

# =========================
# TABLE PROCESSING
# =========================

def process_table(table_item, table_name: str, source: str, doc) -> List[Document]:
    try:
        df = table_item.export_to_dataframe(doc)
    except Exception:
        return []

    docs = []
    columns = list(df.columns)

    for _, row in df.iterrows():
        row_name = str(row.iloc[0])

        values = {}
        for col in columns[1:]:
            values[col] = parse_number(str(row[col]))

        # ===== ROW CHUNK =====
        row_text = f"{row_name}. Значения: " + ", ".join(
            f"{extract_year(col)}: {values[col]}" for col in values
        )

        for part in split_by_tokens(row_text):
            docs.append(Document(
                page_content=part,
                metadata={
                    "type": "table_row",
                    "table": table_name,
                    "metric": row_name,
                    "source": source
                }
            ))

        # ===== CELL CHUNKS =====
        for col in columns[1:]:
            val = values[col]
            year = extract_year(col)

            cell_text = f"{row_name} в {year} году: {val} {DEFAULT_UNIT}"

            for part in split_by_tokens(cell_text):
                docs.append(Document(
                    page_content=part,
                    metadata={
                        "type": "table_cell",
                        "table": table_name,
                        "metric": row_name,
                        "row": row_name,
                        "column": col,
                        "year": year,
                        "value": val,
                        "unit": DEFAULT_UNIT,
                        "source": source
                    }
                ))

    return docs


def extract_tables(doc, source: str) -> List[Document]:
    docs = []

    for i, table in enumerate(doc.tables or []):
        table_name = f"table_{i+1}"
        docs.extend(process_table(table, table_name, source, doc))

    return docs

# =========================
# DEDUP
# =========================

def deduplicate(docs: List[Document]) -> List[Document]:
    seen = set()
    result = []

    for d in docs:
        key = (d.metadata.get("type"), d.page_content[:200])

        if key not in seen:
            seen.add(key)
            result.append(d)

    return result

# =========================
# PIPELINE
# =========================

def process_pdf(path: str) -> List[Document]:
    source = os.path.basename(path)
    print(f"[INFO] Processing {source}")

    all_docs = []

    for doc in convert_pdf_batched(path):
        text_docs = chunk_text(doc, source)
        table_docs = extract_tables(doc, source)

        all_docs.extend(text_docs)
        all_docs.extend(table_docs)

    return deduplicate(all_docs)

# =========================
# RUN
# =========================

if __name__ == "__main__":
    docs = process_pdf("обобщенный отчет сбер за 2025.pdf")

    print(f"Total chunks: {len(docs)}")

    stats = {}
    for d in docs:
        t = d.metadata.get("type")
        stats[t] = stats.get(t, 0) + 1

    print("Stats:", stats)
