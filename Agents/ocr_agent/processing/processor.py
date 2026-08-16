"""OCR processing pipeline."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2

from Agents.ocr_agent.config import OCRConfig
from Agents.ocr_agent.core.correction import TurkishOCRCorrector
from Agents.ocr_agent.core.insights import build_insights, split_lines
from Agents.ocr_agent.core.layout import LayoutAnalyzer
from Agents.ocr_agent.core.ocr_parser import OCRResultParser
from Agents.ocr_agent.core.tables import TableExtractor
from Agents.ocr_agent.document import validate_document
from Agents.ocr_agent.engines.paddle_engine import PaddleStructureEngine
from Agents.ocr_agent.models import OCRProcessingError, PageResult, UnifiedOCRResult
from Agents.ocr_agent.preprocessing.image_preprocessor import ImagePreprocessor
from Agents.ocr_agent.processing.pdf_renderer import PDFRenderer

# If a PDF page already has this much native text, skip heavy OCR for that page.
_NATIVE_TEXT_MIN_CHARS = 40


class OCRProcessor:
    def __init__(self, config: OCRConfig, engine: PaddleStructureEngine | None = None) -> None:
        config.validate()
        self.config = config
        self.engine = engine or PaddleStructureEngine(config)
        self.preprocessor = ImagePreprocessor(config.preprocessing)
        self.pdf_renderer = PDFRenderer(config.pdf_dpi, config.max_pdf_pages)
        self.ocr_parser = OCRResultParser(config.confidence_threshold)
        self.layout_analyzer = LayoutAnalyzer(config.confidence_threshold)
        self.table_extractor = TableExtractor()
        self.corrector = TurkishOCRCorrector()

    def process(self, path: str | Path, document_id: str | None = None) -> UnifiedOCRResult:
        started = perf_counter()
        doc = validate_document(path, self.config.max_file_size_mb)
        # document_id ≠ file name: use provided ID, else stem as stable id
        resolved_id = (document_id or "").strip() or doc.path.stem
        errors: list[OCRProcessingError] = []
        warnings: list[str] = []
        engine_name = getattr(self.engine, "engine_name", "PaddleOCR")

        # Fast path: digital PDF with embedded text
        if doc.is_pdf:
            try:
                native_pages = self.pdf_renderer.extract_text_pages(doc.path)
            except Exception as exc:
                return self._fail(doc, resolved_id, started, str(exc))

            if sum(len(t) for t in native_pages) >= _NATIVE_TEXT_MIN_CHARS:
                pages = []
                for i, text in enumerate(native_pages):
                    if not text:
                        continue
                    corrected = self.corrector.correct(text).text
                    pages.append(
                        PageResult(
                            page_index=i,
                            width=0,
                            height=0,
                            text=corrected,
                            lines=split_lines(corrected),
                            warnings=["extracted_from_pdf_text_layer"],
                        )
                    )
                full_text = "\n\n".join(p.text for p in pages)
                return self._finalize(
                    UnifiedOCRResult(
                        success=bool(full_text.strip()),
                        document_id=resolved_id,
                        file_name=doc.file_name,
                        file_type=doc.extension.lstrip("."),
                        language=self.config.language,
                        pages=pages,
                        full_text=full_text,
                        warnings=["Used embedded PDF text (OCR skipped for digital PDF)."],
                        processing_ms=(perf_counter() - started) * 1000,
                        engine="PyMuPDF text layer",
                    )
                )

        try:
            images = self._load_pages(doc)
        except Exception as exc:
            return self._fail(doc, resolved_id, started, str(exc))

        pages: list[PageResult] = []
        for page_index, image in enumerate(images):
            page_started = perf_counter()
            try:
                processed = (
                    self.preprocessor.process(image)
                    if self.config.preprocessing.enabled
                    else image
                )
                outputs = self.engine.predict(processed)
                raw = self._merge_outputs(outputs)

                text_items = self.ocr_parser.parse(raw, page_index)
                for item in text_items:
                    decision = self.corrector.correct(item.text)
                    item.corrected_text = decision.text
                    item.correction_applied = decision.applied

                layout, visuals = (
                    self.layout_analyzer.analyze(raw, page_index)
                    if self.config.enable_layout
                    else ([], [])
                )
                tables = (
                    self.table_extractor.extract(raw, page_index)
                    if self.config.enable_tables
                    else []
                )
                page_text = "\n".join(item.corrected_text or item.text for item in text_items)
                low_conf = sum(
                    1 for x in text_items if x.confidence < self.config.low_confidence_threshold
                )
                if low_conf:
                    warnings.append(f"Page {page_index}: {low_conf} low-confidence OCR items.")

                pages.append(
                    PageResult(
                        page_index=page_index,
                        width=int(processed.shape[1]),
                        height=int(processed.shape[0]),
                        text=page_text,
                        lines=split_lines(page_text),
                        text_items=text_items,
                        layout=layout,
                        tables=tables,
                        visual_elements=visuals,
                        processing_ms=(perf_counter() - page_started) * 1000,
                    )
                )
                engine_name = getattr(self.engine, "engine_name", engine_name)
            except Exception as exc:
                errors.append(
                    OCRProcessingError(
                        "PAGE_PROCESSING_FAILED",
                        str(exc),
                        page_index=page_index,
                        recoverable=True,
                    )
                )

        full_text = "\n\n".join(p.text for p in pages if p.text)
        success = bool(pages) and bool(full_text.strip())
        if not full_text:
            warnings.append("No readable text was produced from the document.")

        return self._finalize(
            UnifiedOCRResult(
                success=success,
                document_id=resolved_id,
                file_name=doc.file_name,
                file_type=doc.extension.lstrip("."),
                language=self.config.language,
                pages=pages,
                full_text=full_text,
                errors=errors,
                warnings=warnings,
                processing_ms=(perf_counter() - started) * 1000,
                engine=engine_name,
            )
        )

    @staticmethod
    def _finalize(result: UnifiedOCRResult) -> UnifiedOCRResult:
        visuals: list[dict] = []
        for page in result.pages:
            for ve in page.visual_elements:
                visuals.append(
                    {
                        "element_type": ve.element_type,
                        "confidence": ve.confidence,
                        "page_index": ve.page_index,
                    }
                )
            if not page.lines and page.text:
                page.lines = split_lines(page.text)

        insights = build_insights(result.full_text, visuals)
        result.lines = insights.lines
        result.has_signature = insights.has_signature
        result.has_handwritten_signature = insights.has_handwritten_signature
        result.signature_names = insights.signature_names
        result.dates = insights.dates
        result.primary_date = insights.primary_date
        result.has_articles = insights.has_articles
        result.articles = [
            {"number": a.number, "lines": a.lines, "text": a.text} for a in insights.articles
        ]
        if insights.lines:
            result.full_text = "\n".join(insights.lines)
        return result

    def _fail(self, doc, document_id, started, message: str) -> UnifiedOCRResult:
        return UnifiedOCRResult(
            success=False,
            document_id=document_id,
            file_name=doc.file_name,
            file_type=doc.extension.lstrip("."),
            language=self.config.language,
            pages=[],
            full_text="",
            errors=[OCRProcessingError("DOCUMENT_READ_FAILED", message, recoverable=False)],
            processing_ms=(perf_counter() - started) * 1000,
        )

    def _load_pages(self, doc):
        if doc.is_pdf:
            return self.pdf_renderer.render(doc.path)
        image = cv2.imread(str(doc.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {doc.path}")
        return [image]

    @staticmethod
    def _merge_outputs(outputs):
        merged: dict = {}
        for result in outputs:
            raw = PaddleStructureEngine.result_to_dict(result)
            if not raw:
                continue
            for key, value in raw.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(merged[key], list) and isinstance(value, list):
                    merged[key].extend(value)
                elif isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
        return merged
