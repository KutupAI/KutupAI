"""OCR pipeline: validate → pages → quality → OCR → confidence → result."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.core.correction import TurkishOCRCorrector
from Agents.ocr_agent.core.layout import LayoutAnalyzer
from Agents.ocr_agent.core.ocr_parser import OCRResultParser, join_page_text
from Agents.ocr_agent.core.tables import TableExtractor
from Agents.ocr_agent.document import DocumentInput, validate_document
from Agents.ocr_agent.engines.paddle_engine import PaddleStructureEngine, get_shared_engine
from Agents.ocr_agent.exceptions import OCRAgentError, PageExtractionError
from Agents.ocr_agent.interfaces.signature_detector import (
    SealDetection,
    SignatureDetection,
    SignatureSealDetectorInterface,
    VisionHints,
    build_detector,
)
from Agents.ocr_agent.interfaces.vision_fallback import (
    VisionFallbackInterface,
    build_vision_fallback,
    merge_ocr_and_vision,
)
from Agents.ocr_agent.models import LayoutElement, OCRTextItem, TableResult, VisualElement
from Agents.ocr_agent.pipeline import confidence_analyzer, quality_analyzer
from Agents.ocr_agent.preprocessing.image_preprocessor import ImagePreprocessor
from Agents.ocr_agent.processing import office_renderer
from Agents.ocr_agent.processing.pdf_renderer import PDFRenderer
from Agents.ocr_agent.processing.result_builder import (
    build_vision_contract,
    detect_language,
)

logger = logging.getLogger(__name__)

# If a PDF page already has this much native text, skip OCR for that page.
_NATIVE_TEXT_MIN_CHARS = 40


class _PageWork:
    """Internal accumulator for one page's processing across retries."""

    __slots__ = (
        "page_number", "image", "width", "height", "text_items", "layout",
        "visuals", "tables", "attempts", "fallback_used", "warnings",
        "quality_score", "engine_name", "vision_hints",
    )

    def __init__(self, page_number: int, image: np.ndarray | None) -> None:
        self.page_number = page_number
        self.image = image
        self.width = int(image.shape[1]) if image is not None else 0
        self.height = int(image.shape[0]) if image is not None else 0
        self.text_items: list[OCRTextItem] = []
        self.layout: list[LayoutElement] = []
        self.visuals: list[VisualElement] = []
        self.tables: list[TableResult] = []
        self.attempts = 0
        self.fallback_used = False
        self.warnings: list[str] = []
        self.quality_score: float | None = None
        self.engine_name = "n/a"
        self.vision_hints: VisionHints | None = None


class OCRProcessor:
    def __init__(
        self,
        config: OCRConfig,
        engine: PaddleStructureEngine | None = None,
        vision_fallback: VisionFallbackInterface | None = None,
        signature_detector: SignatureSealDetectorInterface | None = None,
    ) -> None:
        config.validate()
        self.config = config
        # Shared/cached engine: heavy model weights load once per process,
        # not once per OCRProcessor/OCRAgent instantiation.
        self.engine = engine or get_shared_engine(config)
        self.preprocessor = ImagePreprocessor(config.preprocessing)
        self.pdf_renderer = PDFRenderer(config.pdf_dpi, config.max_pdf_pages)
        self.ocr_parser = OCRResultParser(config.confidence_threshold)
        self.layout_analyzer = LayoutAnalyzer(config.confidence_threshold)
        self.table_extractor = TableExtractor()
        self.corrector = TurkishOCRCorrector()
        self.vision_fallback = vision_fallback or build_vision_fallback(config.vision_fallback)
        self.signature_detector = signature_detector or build_detector()
        self._fallback_pages_used = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def process(self, path: str | Path, document_id: str | None = None) -> dict[str, Any]:
        started = perf_counter()
        self._fallback_pages_used = 0
        try:
            doc = validate_document(path, self.config.max_file_size_mb)
        except OCRAgentError as exc:
            return self._envelope_failure(
                document_id=document_id, file_name=Path(str(path)).name,
                file_type=Path(str(path)).suffix.lstrip("."), code=exc.code, message=str(exc),
                started=started,
            )

        resolved_id = (document_id or "").strip() or doc.path.stem
        logger.info("ocr_agent: start document_id=%s file=%s", resolved_id, doc.file_name)

        try:
            if doc.is_office:
                page_works = self._process_office(doc)
            elif doc.is_pdf:
                page_works = self._process_pdf(doc)
            else:
                page_works = self._process_image(doc)
        except OCRAgentError as exc:
            return self._envelope_failure(
                document_id=resolved_id, file_name=doc.file_name, file_type=doc.extension.lstrip("."),
                code=exc.code, message=str(exc), started=started,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ocr_agent: unexpected failure for document_id=%s", resolved_id)
            return self._envelope_failure(
                document_id=resolved_id, file_name=doc.file_name, file_type=doc.extension.lstrip("."),
                code="OCR_FAILED", message=str(exc), started=started,
            )

        return self._assemble(doc, resolved_id, page_works, started)

    # ------------------------------------------------------------------
    # Per-format extraction
    # ------------------------------------------------------------------
    def _process_office(self, doc: DocumentInput) -> list[_PageWork]:
        if doc.extension == ".docx":
            texts = office_renderer.extract_docx_pages(doc.path)
        elif doc.extension == ".pptx":
            texts = office_renderer.extract_pptx_pages(doc.path)
        elif doc.extension == ".xlsx":
            texts = office_renderer.extract_xlsx_pages(doc.path)
        else:  # pragma: no cover - guarded by document.py
            raise PageExtractionError(f"Unhandled office extension {doc.extension}")

        works: list[_PageWork] = []
        for i, text in enumerate(texts):
            work = _PageWork(page_number=i + 1, image=None)
            work.engine_name = "native-text-extraction"
            if text.strip():
                corrected = self.corrector.correct(text).text
                work.text_items = [
                    OCRTextItem(text=corrected, confidence=1.0, bounding_box=_zero_bbox(),
                                page_index=i, source="office-text-layer")
                ]
            works.append(work)
        return works

    def _process_pdf(self, doc: DocumentInput) -> list[_PageWork]:
        try:
            native_pages = self.pdf_renderer.extract_text_pages(doc.path)
        except Exception as exc:
            raise PageExtractionError(str(exc)) from exc

        rendered: list[np.ndarray] | None = None
        need_visual = self.config.enable_signature_detection or self.config.enable_visual_elements
        works: list[_PageWork] = []
        for i, native_text in enumerate(native_pages):
            try:
                if len(native_text) >= _NATIVE_TEXT_MIN_CHARS:
                    # Digital page: use the embedded text layer, skip OCR.
                    work = _PageWork(page_number=i + 1, image=None)
                    work.engine_name = "PyMuPDF text layer"
                    corrected = self.corrector.correct(native_text).text
                    work.text_items = [
                        OCRTextItem(text=corrected, confidence=1.0, bounding_box=_zero_bbox(),
                                    page_index=i, source="pdf-text-layer")
                    ]
                    if need_visual:
                        if rendered is None:
                            rendered = self.pdf_renderer.render(doc.path)
                        if i < len(rendered):
                            work.image = rendered[i]
                            work.width = int(work.image.shape[1])
                            work.height = int(work.image.shape[0])
                    works.append(work)
                    continue

                # Scanned/image page inside a (possibly mixed) PDF: render just
                # this page and run the OCR engine on it.
                if rendered is None:
                    rendered = self.pdf_renderer.render(doc.path)
                image = rendered[i] if i < len(rendered) else None
                if image is None:
                    work = _PageWork(page_number=i + 1, image=None)
                    work.warnings.append("Page could not be rasterized.")
                    works.append(work)
                    continue
                works.append(self._ocr_page(i, image))
            except Exception as exc:
                logger.exception("ocr_agent: page %s failed independently", i + 1)
                work = _PageWork(page_number=i + 1, image=None)
                work.warnings.append(f"PAGE_EXTRACTION_FAILED: {exc}")
                works.append(work)
        return works

    def _process_image(self, doc: DocumentInput) -> list[_PageWork]:
        image = cv2.imread(str(doc.path), cv2.IMREAD_COLOR)
        if image is None:
            # OpenCV can fail on WEBP/TIFF/GIF depending on build; try Pillow.
            try:
                from PIL import Image

                with Image.open(doc.path) as pil_img:
                    image = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                raise PageExtractionError(f"Could not decode image: {exc}") from exc
        try:
            return [self._ocr_page(0, image)]
        except Exception as exc:
            logger.exception("ocr_agent: image page failed independently")
            work = _PageWork(page_number=1, image=image)
            work.warnings.append(f"OCR_FAILED: {exc}")
            return [work]

    # ------------------------------------------------------------------
    # Per-page cascade (fast path):
    #   PaddleOCR → Good / usable → accept
    #   empty/failed only → RapidOCR (once, not on every retry)
    #   still empty → PaddleOCR-VL
    #   table detected → PP-StructureV3 Table Pipeline
    # ------------------------------------------------------------------
    def _ocr_page(self, page_index: int, original_image: np.ndarray) -> _PageWork:
        work = _PageWork(page_number=page_index + 1, image=original_image)
        quality = quality_analyzer.assess(original_image, self.config.quality_threshold)
        work.quality_score = quality.score

        image = original_image
        force_full_preprocessing = not quality.readable or quality.poor
        processed = original_image
        fallback_called = False
        accepted = False
        rapid_tried = False

        for attempt in range(1, self.config.max_ocr_attempts + 1):
            work.attempts = attempt
            try:
                if self.config.preprocessing.enabled:
                    if force_full_preprocessing or attempt > 1:
                        processed = self.preprocessor.process(image)
                    else:
                        processed = self.preprocessor.process(image, strength="light")
                else:
                    processed = image
            except Exception as exc:
                work.warnings.append(f"Attempt {attempt} preprocess failed: {exc}")
                if attempt >= self.config.max_ocr_attempts:
                    break
                force_full_preprocessing = True
                continue

            work.width, work.height = int(processed.shape[1]), int(processed.shape[0])

            # --- 1) PaddleOCR (normal text, GPU) ---
            paddle_ok = self._run_engine_pass(
                work, processed, page_index, quality, engine="paddle",
            )
            if paddle_ok:
                accepted = True
                logger.info("[OCR] page=%s engine=PaddleOCR verdict=Good", page_index + 1)
                break

            usable = confidence_analyzer.is_result_usable(
                work.text_items,
                confidence_threshold=self.config.confidence_threshold,
            )
            if usable:
                # Skip RapidCPU cascade when Paddle already returned usable text.
                accepted = True
                work.warnings.append("paddle_usable_skip_rapid")
                logger.info(
                    "[OCR] page=%s engine=PaddleOCR verdict=Usable (skip Rapid)",
                    page_index + 1,
                )
                break

            logger.info("[OCR] page=%s engine=PaddleOCR verdict=Bad", page_index + 1)

            # --- 2) RapidOCR only when Paddle empty/unusable (once) ---
            if (
                self.config.enable_rapid_fallback
                and not rapid_tried
                and not work.text_items
            ):
                rapid_tried = True
                rapid_ok = self._run_engine_pass(
                    work, processed, page_index, quality, engine="rapid",
                )
                if rapid_ok or confidence_analyzer.is_result_usable(
                    work.text_items,
                    confidence_threshold=self.config.confidence_threshold,
                ):
                    accepted = True
                    logger.info(
                        "[OCR] page=%s engine=RapidOCR verdict=%s",
                        page_index + 1,
                        "Good" if rapid_ok else "Usable",
                    )
                    break
                logger.info("[OCR] page=%s engine=RapidOCR verdict=Bad", page_index + 1)

            if attempt < self.config.max_ocr_attempts:
                force_full_preprocessing = True
                continue

            # --- 3) Still empty/bad → PaddleOCR-VL (last resort) ---
            if self.config.vision_fallback.enabled and (
                not work.text_items
                or confidence_analyzer.page_looks_incomplete(
                    processed, work.text_items, quality
                )
            ):
                try:
                    incomplete = confidence_analyzer.page_looks_incomplete(
                        processed, work.text_items, quality
                    )
                    corrupted = confidence_analyzer.page_looks_corrupted(work.text_items)
                    self._try_vision_fallback(
                        work, processed, page_index,
                        incomplete=incomplete or not work.text_items,
                        corrupted=corrupted,
                    )
                    fallback_called = True
                    if work.text_items:
                        accepted = True
                        logger.info(
                            "[OCR] page=%s engine=PaddleOCR-VL verdict=done",
                            page_index + 1,
                        )
                except Exception as exc:
                    work.warnings.append(f"vision_fallback_failed: {exc}")
            break

        # --- 4) Table detected/required → PP-StructureV3 Table Pipeline ---
        table_needed = (
            self.config.enable_tables
            and confidence_analyzer.page_looks_like_table(processed, work.text_items)
        )
        if table_needed:
            logger.info(
                "[OCR] page=%s table detected/required → PP-StructureV3 Table Pipeline",
                page_index + 1,
            )
            try:
                self._run_table_pipeline(work, processed, page_index)
            except Exception as exc:
                work.warnings.append(f"table_pipeline_failed: {exc}")
                logger.warning(
                    "[OCR] page=%s table pipeline Bad: %s", page_index + 1, exc
                )

        if (
            not fallback_called
            and not work.text_items
            and self.config.vision_fallback.enabled
        ):
            try:
                self._try_vision_fallback(
                    work, processed if processed is not None else original_image, page_index,
                    incomplete=True, corrupted=False,
                )
            except Exception as exc:
                work.warnings.append(f"vision_fallback_failed: {exc}")

        if not accepted and not work.text_items:
            work.warnings.append(
                f"Low OCR confidence after {work.attempts} attempt(s); "
                "accepting best-effort result."
            )

        return work

    def _engine_predict(self, image: np.ndarray, engine: str):
        """Call real multi-engine API; FakeEngine tests only accept image."""
        predict = self.engine.predict
        try:
            import inspect

            if "engine" in inspect.signature(predict).parameters:
                return predict(image, engine=engine)
        except (TypeError, ValueError):
            pass
        return predict(image)

    def _run_engine_pass(
        self,
        work: _PageWork,
        processed: np.ndarray,
        page_index: int,
        quality: quality_analyzer.QualityScore,
        *,
        engine: str,
    ) -> bool:
        """Run one OCR engine and apply Good/Bad gate. Returns True if Good."""
        try:
            outputs = self._engine_predict(processed, engine)
            raw = self._merge_outputs(outputs)
        except Exception as exc:
            work.warnings.append(f"{engine} failed: {exc}")
            return False

        work.engine_name = getattr(
            self.engine, "last_engine_name", getattr(self.engine, "engine_name", engine)
        )

        text_items = self.ocr_parser.parse(raw, page_index)
        for item in text_items:
            decision_text = self.corrector.correct(item.text)
            item.corrected_text = decision_text.text
            item.correction_applied = decision_text.applied

        layout, visuals = (
            self.layout_analyzer.analyze(raw, page_index)
            if self.config.enable_layout else ([], [])
        )
        tables = (
            self.table_extractor.extract(raw, page_index)
            if self.config.enable_tables else []
        )

        incomplete = confidence_analyzer.page_looks_incomplete(
            processed, text_items, quality
        )
        corrupted = confidence_analyzer.page_looks_corrupted(text_items)
        good = confidence_analyzer.is_result_good(
            text_items,
            low_confidence_threshold=self.config.low_confidence_threshold,
            incomplete=incomplete,
            corrupted=corrupted,
        )

        # Keep best-so-far even on Bad so later engines / VL can merge.
        if text_items and (
            not work.text_items
            or confidence_analyzer.page_confidence(text_items)
            >= confidence_analyzer.page_confidence(work.text_items)
        ):
            work.text_items = text_items
            work.layout = layout
            work.visuals = visuals
            if tables:
                work.tables = tables

        return good

    def _run_table_pipeline(
        self, work: _PageWork, processed: np.ndarray, page_index: int,
    ) -> None:
        outputs = self._engine_predict(processed, "structure_table")
        raw = self._merge_outputs(outputs)
        tables = self.table_extractor.extract(raw, page_index)
        if tables:
            work.tables = tables
            work.warnings.append("structure_table_pipeline_used")
            logger.info(
                "[OCR] page=%s engine=PP-StructureV3-Table verdict=Good tables=%s",
                page_index + 1, len(tables),
            )
        else:
            work.warnings.append("structure_table_pipeline_no_tables")
            logger.info(
                "[OCR] page=%s engine=PP-StructureV3-Table verdict=Bad", page_index + 1
            )

        # If page text is still empty, harvest any OCR nested in structure output.
        if not work.text_items:
            text_items = self.ocr_parser.parse(raw, page_index)
            for item in text_items:
                decision_text = self.corrector.correct(item.text)
                item.corrected_text = decision_text.text
                item.correction_applied = decision_text.applied
            if text_items:
                work.text_items = text_items
                work.engine_name = getattr(
                    self.engine, "last_engine_name", "PP-StructureV3 Table Pipeline"
                )

        if self.config.enable_layout:
            layout, visuals = self.layout_analyzer.analyze(raw, page_index)
            if layout:
                work.layout = layout
            if visuals:
                work.visuals = visuals

    def _vision_page_budget_exhausted(self) -> bool:
        cap = self.config.vision_fallback.max_pages_per_document
        return cap is not None and self._fallback_pages_used >= cap

    def _try_vision_fallback(
        self,
        work: _PageWork,
        image: np.ndarray,
        page_index: int,
        *,
        incomplete: bool,
        corrupted: bool,
    ) -> None:
        if self._vision_page_budget_exhausted():
            work.warnings.append("vision_fallback_skipped_max_pages")
            return
        logger.info(
            "[OCR] Vision fallback triggered page=%s incomplete=%s corrupted=%s",
            page_index + 1, incomplete, corrupted,
        )
        try:
            result = self.vision_fallback.read_page(image)
        except OCRAgentError as exc:
            work.warnings.append(f"vision_fallback_failed: {exc}")
            return
        except Exception as exc:
            work.warnings.append(f"vision_fallback_failed: {exc}")
            return

        self._fallback_pages_used += 1
        work.vision_hints = VisionHints(
            signature_detected=result.signature_detected,
            signature_handwritten=result.signature_handwritten,
            stamp_detected=result.stamp_detected,
        )
        ocr_text = join_page_text(work.text_items)
        chosen, used_vision = merge_ocr_and_vision(
            ocr_text, result.text, incomplete=incomplete, corrupted=corrupted,
        )
        if result.text.strip() or result.signature_detected or result.stamp_detected:
            work.fallback_used = True
            work.warnings.append("vision_fallback_used")
        if not result.text.strip():
            work.warnings.append("vision_fallback_returned_no_text")
            return
        if not used_vision:
            work.warnings.append("vision_fallback_kept_ocr_text")
            return
        if chosen.strip() == ocr_text.strip():
            work.warnings.append("vision_fallback_verified_ocr_text")
            return
        work.text_items = _items_from_recovered_text(
            chosen,
            confidence=result.confidence if result.confidence is not None else 0.7,
            page_index=page_index,
            source=f"ocr+vision:{result.provider}",
        )
        work.warnings.append("vision_fallback_recovered_regions")

    def _maybe_verify_visuals(self, work: _PageWork, image: np.ndarray | None) -> None:
        """PaddleOCR-VL visual check only when OCR didn't already inspect the page."""
        if image is None or work.vision_hints is not None:
            return
        if not self.config.enable_signature_detection:
            return
        if not self.config.vision_fallback.enabled:
            return
        # Cheap CV first; ask vision only if ink analysis is inconclusive on a
        # page that still looks like it may contain a mark.
        sig, seal = self.signature_detector.detect("", work.visuals, image=image)
        if sig.detected or seal.detected:
            return
        layout_marks = any(
            any(k in (ve.element_type or "").lower() for k in ("image", "figure", "chart"))
            for ve in work.visuals
        )
        if not layout_marks:
            return
        if self._vision_page_budget_exhausted():
            return
        try:
            result = self.vision_fallback.inspect_visuals(image)
        except Exception as exc:
            work.warnings.append(f"vision_verify_failed: {exc}")
            return
        self._fallback_pages_used += 1
        work.fallback_used = True
        work.vision_hints = VisionHints(
            signature_detected=result.signature_detected,
            signature_handwritten=result.signature_handwritten,
            stamp_detected=result.stamp_detected,
        )
        work.warnings.append("vision_verify_used")
    # ------------------------------------------------------------------
    # Assembly into the stable output contract
    # ------------------------------------------------------------------
    def _assemble(
        self, doc: DocumentInput, document_id: str, works: list[_PageWork], started: float,
    ) -> dict[str, Any]:
        pages_out: list[dict[str, Any]] = []
        successful = 0
        full_text_parts: list[str] = []

        for work in works:
            page_text = join_page_text(work.text_items)
            ok = bool(page_text)
            if ok:
                successful += 1
                full_text_parts.append(page_text)
            try:
                signature, seal = self.signature_detector.detect(
                    page_text,
                    work.visuals,
                    image=work.image,
                    vision_hints=work.vision_hints,
                )
            except Exception as exc:
                work.warnings.append(f"SIGNATURE_DETECTION_FAILED: {exc}")
                signature = SignatureDetection(False, False, None, None)
                seal = SealDetection(False, None, None, None)
            vision = build_vision_contract(signature, seal)

            pages_out.append({
                "page_number": work.page_number,
                "text": page_text,
                "vision": vision,
            })

        full_text = "\n\n".join(full_text_parts)
        total_pages = len(works)
        if total_pages == 0 or successful == 0:
            status = "failed"
        elif successful < total_pages:
            status = "partial"
        else:
            status = "complete"
        success = successful > 0

        language = {
            "detected": detect_language(full_text, self.config.language).get(
                "detected", self.config.language
            )
        }

        elapsed_ms = (perf_counter() - started) * 1000
        logger.info(
            "ocr_agent: done document_id=%s status=%s pages=%s/%s duration_ms=%.1f",
            document_id, status, successful, total_pages, elapsed_ms,
        )

        return {
            "success": success,
            "status": status,
            "data": {
                "document_id": document_id,
                "file_name": doc.file_name,
                "file_type": doc.extension.lstrip("."),
                "page_count": total_pages,
                "language": language,
                "pages": pages_out,
                "full_text": full_text,
            },
        }

    def _envelope_failure(
        self, *, document_id: str | None, file_name: str, file_type: str,
        code: str, message: str, started: float,
    ) -> dict[str, Any]:
        logger.error("ocr_agent: failed document_id=%s code=%s message=%s", document_id, code, message)
        return {
            "success": False,
            "status": "failed",
            "error": {"code": code, "message": message},
            "data": {
                "document_id": document_id,
                "file_name": file_name,
                "file_type": file_type,
                "page_count": 0,
                "language": {"detected": None},
                "pages": [],
                "full_text": "",
            },
        }

    @staticmethod
    def _merge_outputs(outputs: Any) -> dict:
        merged: dict = {}
        for result in outputs:
            raw = PaddleStructureEngine.result_to_dict(result)
            if not raw:
                continue
            nested = raw.get("overall_ocr_res")
            if isinstance(nested, dict):
                for key in ("dt_polys", "rec_polys", "rec_boxes", "rec_texts", "rec_scores", "texts", "scores"):
                    if key in nested and key not in raw:
                        raw[key] = nested[key]
            for key, value in raw.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(merged[key], list) and isinstance(value, list):
                    merged[key].extend(value)
                elif isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
        return merged


def _zero_bbox():
    from Agents.ocr_agent.models import BoundingBox

    return BoundingBox([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])


def _items_from_recovered_text(
    text: str, *, confidence: float, page_index: int, source: str,
) -> list[OCRTextItem]:
    """Keep recovered paragraphs as separate OCR blocks (no truncation)."""

    chunks = [part.strip("\n") for part in (text or "").split("\n\n")]
    chunks = [part for part in chunks if part.strip()]
    if not chunks and (text or "").strip():
        chunks = [text.strip()]
    items: list[OCRTextItem] = []
    for part in chunks:
        items.append(
            OCRTextItem(
                text=part,
                confidence=confidence,
                bounding_box=_zero_bbox(),
                page_index=page_index,
                source=source,
                corrected_text=part,
            )
        )
    return items
