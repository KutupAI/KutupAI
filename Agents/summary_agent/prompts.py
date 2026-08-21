"""
Deterministic prompt template for Gemma 3 (Inference/models/gemma3.gguf).

Plain text output — no chain-of-thought — so writer_agent can consume it reliably.
"""

from __future__ import annotations

from typing import List

from .schemas import RAGResultItem

SUMMARY_PROMPT_TEMPLATE = """You are a legal-context summarization step in a government document \
processing pipeline. You do not answer the user. You prepare grounded notes for another \
agent that will draft the official response.

Instructions:
1. Summarize only the parts of CONTEXT relevant to QUESTION.
2. Use only CONTEXT. Never add outside knowledge or invented facts.
3. Preserve exactly as written: law numbers, article numbers, dates, conditions, and exceptions.
4. Drop information that is irrelevant to QUESTION or duplicated across sources.
5. After each preserved point, tag its source in the form [law_number/article_no].
6. Do not address the user, give an opinion, or answer the question directly.
7. If nothing in CONTEXT is relevant to QUESTION, output exactly: "no_relevant_context".
8. Output plain text bullet points ("- "), one point per line, no headers, no markdown, no preamble.

QUESTION:
{question}

CONTEXT:
{rag_results}

SUMMARY:"""


def _format_chunk(index: int, item: RAGResultItem) -> str:
    ref = f"{item.law_number or '?'}/{item.article_no or '?'}"
    pages = (
        f"pp.{item.page_start}-{item.page_end}"
        if item.page_start is not None and item.page_end is not None
        else ""
    )
    header = f"[{index}] {ref} ({item.article_type or 'n/a'}) {pages}".strip()
    return f"{header}\n{item.text.strip()}"


def build_prompt(question: str, results: List[RAGResultItem]) -> str:
    """Render the deterministic summary prompt from validated RAG chunks."""
    rag_results = "\n\n".join(_format_chunk(i, r) for i, r in enumerate(results, start=1))
    return SUMMARY_PROMPT_TEMPLATE.format(question=question.strip(), rag_results=rag_results)
