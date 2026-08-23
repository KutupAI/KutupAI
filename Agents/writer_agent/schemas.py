"""
schemas.py -- Internal data contracts for the Writer Agent.

These dataclasses describe:
  * WriterContext  -- the compact context assembled from the Unified
                       State, which is what actually gets sent to
                       Gemma3 (never the full Unified State).
  * WritingResult  -- the shape of the `writing` section the agent
                       writes back into the Unified State.

They do NOT replace or redefine the project's Unified State schema.
They exist only to type what the Writer Agent itself consumes and
produces internally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class WriterContext:
    """Compact, token-minimized context handed to the model."""

    question: str
    document_type: str = ""
    summary: str = ""
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WritingResult:
    """Matches the `writing` section of the Unified State."""

    success: bool
    answer: str = ""
