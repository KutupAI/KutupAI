"""Classification Agent package."""

from Agents.classification_agent.agent import (
    CLASSIFICATION_CONTRACT_KEYS,
    ClassificationAgent,
    process,
)
from Agents.classification_agent.config import ClassificationConfig
from Agents.classification_agent.models import ClassificationAlternative, ClassificationResult
from Agents.classification_agent.taxonomy import DOCUMENT_CLASSES

__all__ = [
    "CLASSIFICATION_CONTRACT_KEYS",
    "ClassificationAgent",
    "ClassificationConfig",
    "ClassificationAlternative",
    "ClassificationResult",
    "DOCUMENT_CLASSES",
    "process",
]
