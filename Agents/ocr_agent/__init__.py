"""OCR Agent package.

Public surface for Orchestration/tests:
    OCRAgent      - BaseAgent entry point (state in, state out)
    OCRClient     - stable client used by the agent (and directly by tests/tools)
    OCRRequest    - client request dataclass
    OCRConfig     - externalized configuration (env-driven)
    OCRProcessor  - the pipeline itself, for advanced/direct use or testing
"""

from Agents.ocr_agent.agent import OCRAgent
from Agents.ocr_agent.client import OCRClient, OCRRequest
from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.processing.processor import OCRProcessor

__all__ = [
    "OCRAgent",
    "OCRClient",
    "OCRRequest",
    "OCRConfig",
    "OCRProcessor",
]
