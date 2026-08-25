"""Writer Agent: produces the final answer in the Unified State contract."""

from __future__ import annotations

from typing import Any, Dict, Optional

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent
from Inference.client.inference_request import InferenceRequest, Message
from Inference.client.llama_client import LlamaClient

from .config import MAX_OCR_CHARS, MAX_SUMMARY_CHARS, MAX_TOKENS, TEMPERATURE, TOP_P
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import WriterContext, WritingResult


@register
class WriterAgent(BaseAgent):
    """Create ``state["writing"]`` without modifying other state sections."""

    name = "writer_agent"
    description = "Generate the final, source-grounded response for the document question."

    def __init__(self, client: Optional[LlamaClient] = None) -> None:
        self.client = client or LlamaClient()

    def _extract_ocr_text(self, state: Dict[str, Any]) -> str:
        ocr = state.get("ocr") or {}
        ocr_data = ocr.get("ocr_data") if isinstance(ocr, dict) else {}
        if isinstance(ocr_data, dict):
            text = str(ocr_data.get("full_text") or "")
            if text.strip():
                return text[:MAX_OCR_CHARS]
        text = str(state.get("document_text") or "")
        if text.strip():
            return text[:MAX_OCR_CHARS]
        ocr_result = state.get("ocr_result") or {}
        if isinstance(ocr_result, dict):
            data = ocr_result.get("Data") or ocr_result.get("data") or []
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return str(data[0].get("full_text") or "")[:MAX_OCR_CHARS]
        return ""

    def _extract_context(self, state: Dict[str, Any]) -> WriterContext:
        request = state.get("request") or {}
        classification = state.get("classification") or state.get("classification_result") or {}
        extraction = state.get("extraction") or state.get("extraction_result") or {}
        validation = state.get("validation") or state.get("validation_result") or {}
        summary = state.get("summary") or {}
        summary_data = summary.get("data") if isinstance(summary.get("data"), dict) else {}
        summary_text = (
            summary.get("rag_summary_text")
            or summary.get("summary_text")
            or summary.get("summary")
            or summary.get("text")
            or summary_data.get("summary")
            or state.get("summary_text")
            or ""
        )
        summary_text = str(summary_text)[:MAX_SUMMARY_CHARS]

        extracted_data = {
            key: value
            for key, value in extraction.items()
            if key != "success" and value not in (None, "", [], {})
        }
        validation_info: Dict[str, Any] = {}
        if validation.get("is_complete") is False:
            validation_info["is_complete"] = False
        if validation.get("errors"):
            validation_info["errors"] = validation["errors"]
        if validation.get("warnings"):
            validation_info["warnings"] = validation["warnings"]
        extracted_data.update(validation_info)

        return WriterContext(
            question=request.get("question") or state.get("question") or "",
            document_type=classification.get("document_type") or classification.get("doc_type") or "",
            summary=summary_text,
            document_text=self._extract_ocr_text(state),
            extracted_data=extracted_data,
            validation=validation_info,
        )

    def _build_messages(self, context: WriterContext) -> list[Message]:
        return [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=build_user_prompt(
                    question=context.question,
                    document_type=context.document_type,
                    summary=context.summary,
                    extracted_data=context.extracted_data,
                    document_text=context.document_text,
                ),
            ),
        ]

    def _call_inference(self, messages: list[Message]) -> WritingResult:
        response = self.client.generate(
            InferenceRequest(
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
        )
        if not response.success or not response.text:
            return WritingResult(success=False, answer="")
        return WritingResult(success=True, answer=response.text.strip())

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(state, dict):
            raise TypeError("WriterAgent.run expects the Unified State as a dict")

        updated = dict(state)
        try:
            context = self._extract_context(updated)
            if not context.question:
                updated["writing"] = {"success": False, "answer": ""}
                return updated
            result = self._call_inference(self._build_messages(context))
            updated["writing"] = {"success": result.success, "answer": result.answer}
        except Exception:
            updated["writing"] = {"success": False, "answer": ""}
        return updated
