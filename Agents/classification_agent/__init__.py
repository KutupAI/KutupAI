"""Classification Agent package."""

from Agents.classification_agent.agent import ClassificationAgent
from Agents.classification_agent.config import ClassificationConfig
from Agents.classification_agent.models import ClassificationAlternative, ClassificationResult
from Agents.classification_agent.taxonomy import DOCUMENT_CLASSES

__all__ = [
    "ClassificationAgent",
    "ClassificationConfig",
    "ClassificationAlternative",
    "ClassificationResult",
    "DOCUMENT_CLASSES",
]
