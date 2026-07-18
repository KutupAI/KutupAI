"""
indexer.py
------------
خط أنابيب الفهرسة الكامل: تقطيع → metadata → embedding → حفظ.

هذا هو الملف الذي يُشغَّل فعليًا لفهرسة مستند قانوني (أو مجلد كامل
من المستندات) في ChromaDB. يجمع بين:
  chunker.py -> metadata_extractor.py -> embedding_model.py -> chroma_store.py
"""

import hashlib
from pathlib import Path
from typing import List

from RAG.configuration.rag_config_loader import documents_config, indexing_config
from RAG.embeddings.embedding_model import embed_batch
from RAG.indexing.chunker import chunk_document
from RAG.indexing.metadata_extractor import EnrichedChunk, enrich_chunks
from RAG.vector_store.chroma_store import get_vector_store


def _make_chunk_id(source_file: Path, article_number: str, part_index: int) -> str:
    """
    توليد معرّف فريد وثابت لكل chunk (نفس المدخلات تعطي دائمًا نفس الـ id).
    هذا مهم لعملية إعادة الفهرسة (update_index.py) حتى لا تتكرر السجلات.
    """
    raw = f"{source_file.name}:{article_number}:{part_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def index_file(file_path: Path) -> int:
    """
    فهرسة ملف قانون واحد بالكامل.

    Args:
        file_path: مسار ملف النص المصدر (.txt/.md).

    Returns:
        عدد الـ chunks التي تمت فهرستها من هذا الملف.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    raw_chunks = chunk_document(full_text)
    enriched_chunks: List[EnrichedChunk] = enrich_chunks(raw_chunks, file_path)

    if not enriched_chunks:
        return 0

    texts = [c.text for c in enriched_chunks]
    metadatas = [c.metadata for c in enriched_chunks]
    ids = [
        _make_chunk_id(file_path, m["article_number"], m["part_index"])
        for m in metadatas
    ]

    embeddings = embed_batch(texts)

    store = get_vector_store()
    store.add_documents(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

    return len(enriched_chunks)


def index_directory(directory_path: Path) -> dict:
    """
    فهرسة كل الملفات المسموحة داخل مجلد (مثال: RAG/documents/laws).

    Args:
        directory_path: مسار المجلد.

    Returns:
        قاموس {اسم_الملف: عدد_الـ_chunks} كملخص لعملية الفهرسة.
    """
    summary: dict = {}
    allowed_ext = set(documents_config.allowed_extensions)

    for file_path in sorted(directory_path.glob("*")):
        if file_path.suffix.lower() not in allowed_ext:
            continue
        if file_path.name.endswith(".meta.json"):
            continue

        count = index_file(file_path)
        summary[file_path.name] = count

    return summary


def index_all_sources() -> dict:
    """
    فهرسة كل مصادر الوثائق المعرّفة في rag_config.yaml
    (laws + regulations + internal_docs) دفعة واحدة.

    Returns:
        ملخص شامل لكل الملفات المفهرسة عبر كل المسارات.
    """
    full_summary: dict = {}
    for path_str in [
        documents_config.laws_path,
        documents_config.regulations_path,
        documents_config.internal_docs_path,
    ]:
        directory = Path(path_str)
        if directory.exists():
            full_summary.update(index_directory(directory))

    return full_summary


if __name__ == "__main__":
    result = index_all_sources()
    total_chunks = sum(result.values())
    print(f"تمت فهرسة {len(result)} ملف/ملفات، بإجمالي {total_chunks} chunk.")
    for filename, count in result.items():
        print(f"  - {filename}: {count} chunk")
