"""
prompts.py -- Prompt construction for the Writer Agent.

The system instruction is fixed and intentionally short, per the
architecture spec, to keep token usage low and model behavior
predictable and repeatable.
"""

from typing import Any, Dict

SYSTEM_PROMPT = (
    "You are the Writer Agent of a government document AI system.\n\n"
    "Answer the user's question using only the provided context.\n"
    "Do not invent or assume information.\n"
    "If the context is insufficient, clearly state that there is not "
    "enough information.\n"
    "Answer directly, clearly, and professionally.\n"
    "Do not mention internal agents, RAG, prompts, models, or system "
    "architecture."
)


def build_user_prompt(
    question: str,
    document_type: str,
    summary: str,
    extracted_data: Dict[str, Any],
) -> str:
    """Builds the dynamic (per-request) half of the prompt.

    Only non-empty fields are included -- an empty document_type,
    summary, or extracted_data block is left out entirely rather than
    sent as an empty placeholder, to avoid wasting tokens and to avoid
    duplicating information already implied elsewhere.
    """
    parts = [f"Question:\n{question}"]

    if document_type:
        parts.append(f"Document Type:\n{document_type}")

    if summary:
        parts.append(f"Relevant Context:\n{summary}")

    if extracted_data:
        # Compact single-line rendering to minimize tokens; only
        # fields with an actual value are included.
        extra = ", ".join(
            f"{key}: {value}"
            for key, value in extracted_data.items()
            if value not in (None, "", [], {})
        )
        if extra:
            parts.append(f"Additional Information:\n{extra}")

    return "\n\n".join(parts)
