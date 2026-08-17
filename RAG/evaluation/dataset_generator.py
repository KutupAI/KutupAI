"""Generate eval dataset from indexed chunks (LLM if available, else template)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from RAG.configuration.rag_config_loader import evaluation_config
from RAG.vector_store.chroma_store import get_vector_store


def _template_questions(text: str, meta: dict) -> List[str]:
    """
    KutupAI Realistic Dataset Generator:
    Gerçek dünya senaryolarını yansıtmak için hem spesifik (madde numarası içeren) 
    hem de genel semantik (sadece metne dayalı) sorular üretir.
    """
    law_name = meta.get("law_name", "belge")
    law_number = str(meta.get("law_number", ""))
    article = str(meta.get("article_number") or meta.get("article_no", "?"))
    snippet = " ".join((text or "").split())[:150]
    
    questions = []
    
    # 1. Spesifik Soru Tipi (Classification Agent'ı test eder - %30 ihtimalle)
    # Kullanıcı bazen tam kanun ve madde numarasıyla arama yapar.
    if law_number and law_number != "unknown":
        questions.append(f"{law_number} sayılı {law_name} Madde {article} ne düzenler?")
    
    # 2. Genel Semantik Soru Tipi (Hybrid Search ve BGE Reranker'ı test eder - Gerçekçi)
    # Kullanıcı kanun numarası bilmeden sadece konuyu sorar.
    questions.append(f"{law_name} kapsamında {snippet} ... ile ilgili hükümler nelerdir?")
    
    # 3. Doğal Dil Soru Tipi (En gerçekçi senaryo)
    questions.append(f"{snippet} ... Buna göre ilgili mevzuatta nasıl bir düzenleme yapılmıştır?")
    
    # Konfigürasyondaki maksimum soru sayısına göre kırp ve döndür
    return questions[: evaluation_config.max_questions_per_chunk]


def _llm_questions(text: str, meta: dict) -> Optional[List[str]]:
    try:
        from Inference.client.inference_request import InferenceRequest, Message
        from Inference.client.llama_client import LlamaClient
    except Exception:
        return None

    prompt = (
        "Aşağıdaki hukuki metinden Türkçe 1-2 kısa soru üret. Her satırda bir soru.\n\n"
        f"Kaynak: {meta.get('law_name')} Madde {meta.get('article_number')}\n"
        f"Metin:\n{text[:1200]}"
    )
    response = LlamaClient().generate(
        InferenceRequest(
            messages=[
                Message(role="system", content="Hukuki evaluation dataset üreticisisin."),
                Message(role="user", content=prompt),
            ],
            temperature=evaluation_config.llm_temperature,
            max_tokens=evaluation_config.llm_max_tokens,
        )
    )
    if not response.success:
        return None
    lines = [ln.strip("-• ").strip() for ln in response.text.splitlines() if ln.strip()]
    qs = [ln for ln in lines if "?" in ln or len(ln) > 12]
    return qs[: evaluation_config.max_questions_per_chunk] or None


def generate_eval_dataset(
    *,
    output_path: Path | str | None = None,
    max_chunks: int = 50,
    use_llm: bool = True,
    include_preamble: bool = False,
) -> Path:
    output_path = Path(output_path or evaluation_config.default_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples: List[Dict[str, Any]] = []
    indexed_chunks = 0
    for row in get_vector_store().export_all():
        meta = row.get("metadata") or {}
        article = meta.get("article_number") or meta.get("article_no")
        if not include_preamble and article in (None, "", "unknown"):
            continue
        if indexed_chunks >= max_chunks:
            break
        indexed_chunks += 1
        questions = (_llm_questions(row.get("text") or "", meta) if use_llm else None) or _template_questions(
            row.get("text") or "", meta
        )
        for q in questions:
            examples.append(
                {
                    "query": q,
                    "relevant_chunk_ids": [str(row.get("chunk_id"))],
                    "relevant_source_files": [str(meta.get("source_file", ""))],
                    "notes": f"auto article={meta.get('article_number')}",
                    "split": "synthetic",
                }
            )

    payload = {"version": 1, "generator": "llm+template" if use_llm else "template", "examples": examples}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-chunks", type=int, default=50)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--include-preamble", action="store_true")
    args = parser.parse_args()
    print(f"Wrote: {generate_eval_dataset(output_path=args.output, max_chunks=args.max_chunks, use_llm=not args.no_llm, include_preamble=args.include_preamble)}")
