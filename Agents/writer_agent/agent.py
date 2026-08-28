"""Writer Agent: produces the final answer in the Unified State contract."""

from __future__ import annotations

import calendar
from datetime import date
import re
from typing import Any, Dict, Optional

from Agents.base.agent_registry import register
from Agents.base.base_agent import BaseAgent
from Inference.client.evren_client import EvrenClient
from Inference.client.inference_request import InferenceRequest, Message
from Inference.client.llama_client import LlamaClient

from .config import (
    EVREN_MODEL,
    INFERENCE_BACKEND,
    MAX_OCR_CHARS,
    MAX_SUMMARY_CHARS,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import WriterContext, WritingResult


_CONTEXT_REJECTION_MARKERS = (
    "not enough information",
    "does not include the specific",
    "cannot provide",
    "bilgi bulunmamaktadır",
    "yeterli bilgi bulunmamaktadır",
)
_DATE_PATTERN = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")
_CALENDAR_DIFFERENCE_PATTERN = re.compile(
    r"\b(?:kaç|kac)\s+yıl[,\s]+ay\s+ve\s+gün.*\b(?:yürürlüğe|yururluge)\b",
    re.IGNORECASE,
)
_BOLD_PATTERN = re.compile(r"\*{2,3}(.+?)\*{2,3}", re.DOTALL)
_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_LINE_BULLET_PATTERN = re.compile(r"^[ \t]*[*•]+[ \t]+", re.MULTILINE)
_INLINE_BULLET_PATTERN = re.compile(r"[ \t]*[*•]+[ \t]+")
_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


@register
class WriterAgent(BaseAgent):
    """Create ``state["writing"]`` without modifying other state sections."""

    name = "writer_agent"
    description = "Generate the final, source-grounded response for the document question."

    def __init__(self, client: Optional[Any] = None) -> None:
        if client is not None:
            self.client = client
        elif INFERENCE_BACKEND == "evren":
            self.client = EvrenClient(model=EVREN_MODEL)
        else:
            self.client = LlamaClient()

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
            question=(
                request.get("question")
                or state.get("question")
                or state.get("writer_instruction")
                or ""
            ),
            document_type=classification.get("document_type") or classification.get("doc_type") or "",
            summary=summary_text,
            document_text=self._extract_ocr_text(state),
            extracted_data=extracted_data,
            validation=validation_info,
            conversation_memory=str(state.get("conversation_memory") or "")[:1800],
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
                    conversation_memory=context.conversation_memory,
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

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        """Markdown artıklarını temizler ve satır içi maddeleri listeye çevirir."""
        text = answer.replace("\r\n", "\n").replace("`", "")
        text = _BOLD_PATTERN.sub(r"\1", text)
        text = _HEADING_PATTERN.sub("", text)
        text = _LINE_BULLET_PATTERN.sub("- ", text)
        text = _INLINE_BULLET_PATTERN.sub("\n- ", text)
        cleaned: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped in ("-", "*"):
                continue
            if stripped.startswith("- "):
                item = stripped[2:].strip().rstrip(",;")
                if not item:
                    continue
                # Maddeler arasında boş satır bırakılmaz.
                if cleaned and not cleaned[-1]:
                    cleaned.pop()
                cleaned.append(f"- {item}")
            else:
                cleaned.append(stripped)
        text = "\n".join(cleaned)
        text = _BLANK_LINES_PATTERN.sub("\n\n", text)
        return text.strip()

    @staticmethod
    def _rejects_available_context(answer: str) -> bool:
        normalized = answer.casefold()
        return any(marker in normalized for marker in _CONTEXT_REJECTION_MARKERS)

    @staticmethod
    def _calendar_difference(start: date, end: date) -> tuple[int, int, int]:
        years, months, days = end.year - start.year, end.month - start.month, end.day - start.day
        if days < 0:
            months -= 1
            previous_month = end.month - 1 or 12
            previous_year = end.year if end.month > 1 else end.year - 1
            days += calendar.monthrange(previous_year, previous_month)[1]
        if months < 0:
            years -= 1
            months += 12
        return years, months, days

    @classmethod
    def _date_difference_answer(cls, context: WriterContext) -> str:
        """Açık takvim farkı sorularında model yerine doğrulanabilir hesap yapar."""
        if not _CALENDAR_DIFFERENCE_PATTERN.search(context.question):
            return ""
        dates: set[date] = set()
        for day, month, year in _DATE_PATTERN.findall(f"{context.summary}\n{context.conversation_memory}"):
            try:
                dates.add(date(int(year), int(month), int(day)))
            except ValueError:
                continue
        if len(dates) < 2:
            return ""
        start, end = min(dates), max(dates)
        years, months, days = cls._calendar_difference(start, end)
        return (
            f"Yürürlük tarihleri {start.strftime('%d/%m/%Y')} ve {end.strftime('%d/%m/%Y')}'dir. "
            f"Takvim hesabıyla fark {years} yıl {months} ay {days} gündür."
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(state, dict):
            raise TypeError("WriterAgent.run expects the Unified State as a dict")

        updated = dict(state)
        try:
            context = self._extract_context(updated)
            if not context.question:
                updated["writing"] = {"success": False, "answer": ""}
                return updated
            calculated_answer = self._date_difference_answer(context)
            result = (
                WritingResult(success=True, answer=calculated_answer)
                if calculated_answer
                else self._call_inference(self._build_messages(context))
            )
            # Küçük yerel model bazen açık RAG özetini görmezden gelip bağlamın
            # yetersiz olduğunu söylüyor. Böyle bir durumda kanıtlı özeti koru.
            if result.success and context.summary and self._rejects_available_context(result.answer):
                result = WritingResult(success=True, answer=context.summary)
            updated["writing"] = {
                "success": result.success,
                "answer": self._normalize_answer(result.answer),
            }
        except Exception:
            updated["writing"] = {"success": False, "answer": ""}
        return updated
