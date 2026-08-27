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
    document_text: str = "",
    conversation_memory: str = "",
) -> str:
    """Builds the dynamic (per-request) half of the prompt.

    Preference order for grounding text: summary (RAG) → OCR document_text.
    Only non-empty fields are included.
    """
    parts = [f"Question:\n{question}"]

    if document_type:
        parts.append(f"Document Type:\n{document_type}")

    if summary:
        parts.append(f"Relevant Context:\n{summary}")
    elif document_text:
        parts.append(f"Document Text:\n{document_text}")

    if extracted_data:
        extra = ", ".join(
            f"{key}: {value}"
            for key, value in extracted_data.items()
            if value not in (None, "", [], {})
        )
        if extra:
            parts.append(f"Additional Information:\n{extra}")

    if conversation_memory:
        parts.append(
            "Conversation Reference (use only to resolve references such as 'this law'; "
            "it is not legal evidence):\n" + conversation_memory
        )

    return "\n\n".join(parts)
