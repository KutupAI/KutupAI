"""
Mock Agents used exclusively by Orchestration's own test suite.

These are NOT real Agent implementations and are never used outside
`Orchestration/tests/` (see brief: "Mocks are allowed only for
Orchestration tests" / "Do not create fake Agent implementations").
Each mock only implements the `run(state) -> state` contract real Agents
are expected to expose.
"""

from __future__ import annotations

from typing import Any, Dict


class MockOCRAgent:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        if self.fail:
            state["ocr_status"] = "failed"
            state["ocr_result"] = {"Success": False, "Data": []}
            return state
        state["ocr_status"] = "completed"
        state["ocr_result"] = {
            "Success": True,
            "Data": [{"document_id": state.get("document_id"), "full_text": "hello world"}],
        }
        return state


class MockClassificationAgent:
    def __init__(self, *, requires_rag: bool = False) -> None:
        self.requires_rag = requires_rag

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["classification_status"] = "completed"
        state["classification_result"] = {"doc_type": "invoice", "requires_rag": self.requires_rag}
        return state


class MockExtractionAgent:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["extraction_status"] = "completed"
        state["extraction_result"] = {"fields": {"amount": 100}}
        return state


class MockValidationAgent:
    def __init__(self, *, requires_rag: bool = True) -> None:
        self.requires_rag = requires_rag

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["validation_status"] = "completed"
        state["validation_result"] = {"valid": True, "requires_rag": self.requires_rag}
        return state


class MockRagAgent:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = (state.get("request") or {}).get("question") or state.get("question") or ""
        state["rag_status"] = "completed"
        # Shape matches RAGResult / what SummaryAgent adapts from state["rag_result"].
        state["rag_result"] = {
            "success": True,
            "data": {
                "operation": "retrieve",
                "query": question,
                "results": [
                    {
                        "chunk_id": "chunk-a",
                        "text": "context A",
                        "law_number": "6458",
                        "article_no": "31",
                    },
                    {
                        "chunk_id": "chunk-b",
                        "text": "context B",
                        "law_number": "6458",
                        "article_no": "32",
                    },
                ],
            },
            "error": None,
        }
        return state


class MockSummaryAgent:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Matches real SummaryAgent → writer/routing contract.
        state["summary"] = {"success": True, "rag_summary_text": "summary text"}
        return state


class MockRoutingAgent:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Matches real RoutingAgent contract: state["routing"].
        state["routing"] = {"success": True, "department": "finance"}
        return state


class MockWriterAgent:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["writing"] = {"success": True, "answer": "Dear Sir, ..."}
        return state


class AlwaysFailingAgent:
    """Simulates an Agent whose stage keeps failing (e.g. reports
    status=failed) so retry/fallback/termination policies can be
    exercised."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.calls = 0

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        state[f"{self.stage}_status"] = "failed"
        return state


class ExceptionAgent:
    """Simulates an Agent that raises instead of returning a result."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.calls = 0

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.calls += 1
        raise RuntimeError(f"{self.stage} boom")


class InvalidResultAgent:
    """Simulates a badly-behaved Agent that doesn't return a dict."""

    def run(self, state: Dict[str, Any]):  # type: ignore[override]
        return "not-a-dict"
