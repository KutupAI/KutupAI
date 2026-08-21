"""
Manual runner — Mock RAG -> summary_agent -> llama-server (Gemma 3) -> JSON.

Prerequisites:
    1. Gemma 3 loaded via Inference/llamastart.bat (port 8080).
    2. pip install -r Agents/summary_agent/requirements.txt

Usage (from project root):
    python -m Agents.summary_agent.manual_test
    INFERENCE_URL=http://127.0.0.1:8080/v1/chat/completions python -m Agents.summary_agent.manual_test
"""

from __future__ import annotations

import json

from .agent import SummaryAgent
from .mock_data import MOCK_QUESTION, MOCK_RAG_RESULT


def main() -> None:
    agent = SummaryAgent()
    result = agent.summarize(MOCK_QUESTION, MOCK_RAG_RESULT)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
