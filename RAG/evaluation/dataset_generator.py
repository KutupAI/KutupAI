"""Generate eval dataset from indexed chunks (LLM if available, else template)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from RAG.configuration.rag_config_loader import evaluation_config
from RAG.vector_store.chroma_store import get_vector_store


def _template_questions(text: str, meta: dict) -> List[str]:
    law = meta.get("law_name", "belge")
    article = meta.get("article_number", "?")
    snippet = " ".join((text or "").split())[:120]
    return [
        f"{law} Madde {article} ne düzenler?",
        f"{snippet} ... ile ilgili hüküm nedir?",
    ][: evaluation_config.max_questions_per_chunk]


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
) -> Path:
    output_path = Path(output_path or evaluation_config.default_dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples: List[Dict[str, Any]] = []
    for row in get_vector_store().export_all()[:max_chunks]:
        meta = row.get("metadata") or {}
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
    args = parser.parse_args()
    print(f"Wrote: {generate_eval_dataset(output_path=args.output, max_chunks=args.max_chunks, use_llm=not args.no_llm)}")
