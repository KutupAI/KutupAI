"""OCR engine backends."""

from Agents.ocr_agent.engines.paddle_engine import PaddleStructureEngine, get_shared_engine

__all__ = ["PaddleStructureEngine", "get_shared_engine"]
