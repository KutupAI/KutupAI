"""
--------
Extraction_agent tarafından ihtiyaç duyulan harici araçlar/entegrasyonlar.

Tasarım prensibi: Buradaki HER araç hata toleranslıdır. Eksik bir API anahtarı,
indirilemeyen bir model veya hatalı bir LLM yanıtı asla
ajanı çökertemez - sorunsuz bir şekilde çalışmalıdır (boş sonuç + ExtractionMeta'da uyarı) böylece işlem hattının geri kalanı çalışmaya devam eder. Bu,
ağ/model kullanılabilirliğinin tahmin edilemez olduğu test/demo gününde en çok önem taşır.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from .config import DEFAULT_CONFIG, ExtractionAgentConfig
from .models import FieldValue
from .prompts import (
    SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    VISION_USER_PROMPT,
    build_extraction_prompt,
)

logger = logging.getLogger("extraction_agent.tools")


# Shared helpers
def safe_json_loads(text: str) -> Optional[dict[str, Any]]:
    """Parse JSON that may be wrapped in ```json fences or have stray text."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # last resort: grab the outermost {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


# 1. Rule-Based Extraction (Regex) - report section 3.1
class RegexExtractor:
    """Fast, deterministic extraction for structured fields."""

    def __init__(self, cfg: ExtractionAgentConfig = DEFAULT_CONFIG):
        self.cfg = cfg.regex

    def extract_dates(self, text: str) -> list[FieldValue]:
        results: list[FieldValue] = []
        for pattern in self.cfg.date_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                results.append(FieldValue(value=m.group(0), confidence=0.95, source="regex"))
        return results

    def extract_phones(self, text: str) -> list[FieldValue]:
        results: list[FieldValue] = []
        seen = set()
        for pattern in self.cfg.phone_patterns:
            for m in re.finditer(pattern, text):
                digits = re.sub(r"\D", "", m.group(0))
                if len(digits) < 10 or digits in seen:
                    continue
                seen.add(digits)
                results.append(FieldValue(value=m.group(0).strip(), confidence=0.9, source="regex"))
        return results

    def extract_emails(self, text: str) -> list[FieldValue]:
        return [
            FieldValue(value=m.group(0), confidence=0.97, source="regex")
            for m in re.finditer(self.cfg.email_pattern, text)
        ]

    def extract_evrak_no(self, text: str) -> Optional[FieldValue]:
        for pattern in self.cfg.evrak_no_patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                value = m.group(1) if m.groups() else m.group(0)
                return FieldValue(value=value.strip(), confidence=0.85, source="regex")
        return None

    def extract_all(self, text: str) -> dict[str, Any]:
        if not text:
            return {"dates": [], "phones": [], "emails": [], "evrak_no": None}
        return {
            "dates": self.extract_dates(text),
            "phones": self.extract_phones(text),
            "emails": self.extract_emails(text),
            "evrak_no": self.extract_evrak_no(text),
        }


# 2. NLP Entity Extraction (NER) - report section 3.2
class NEREngine:
    """
    Lazy-loaded HuggingFace NER pipeline (BERTurk / ModernBERT-TR family).

    The model is only loaded on first real use (not at import time), and
    any failure to load (no internet, no GPU, package missing) degrades
    to an empty result with a warning instead of raising.
    """

    def __init__(self, cfg: ExtractionAgentConfig = DEFAULT_CONFIG):
        self.cfg = cfg.ner
        self._pipeline = None
        self._load_error: Optional[str] = None
        self._attempted = False

    def _ensure_loaded(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._attempted:
            return False
        self._attempted = True
        if not self.cfg.enabled:
            self._load_error = "NER devre disi (config.ner.enabled=False)"
            return False
        try:
            from transformers import pipeline  # local import: heavy dependency

            self._pipeline = pipeline(
                "ner",
                model=self.cfg.model_name,
                aggregation_strategy=self.cfg.aggregation_strategy,
                device=0 if self.cfg.device.startswith("cuda") else -1,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - must never crash the agent
            self._load_error = f"NER modeli yuklenemedi ({self.cfg.model_name}): {exc}"
            logger.warning(self._load_error)
            return False

    def extract_entities(self, text: str) -> dict[str, list[FieldValue]]:
        result: dict[str, list[FieldValue]] = {"person": [], "organization": [], "location": []}
        if not text:
            return result
        if not self._ensure_loaded():
            return result

        snippet = text[: self.cfg.max_chars]
        try:
            entities = self._pipeline(snippet)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NER inference basarisiz: %s", exc)
            return result

        label_map = {
            "PER": "person",
            "PERSON": "person",
            "ORG": "organization",
            "LOC": "location",
            "GPE": "location",
        }
        for ent in entities:
            group = label_map.get(str(ent.get("entity_group", "")).upper())
            if not group:
                continue
            result[group].append(
                FieldValue(
                    value=ent.get("word", "").strip(),
                    confidence=float(ent.get("score", 0.0)),
                    source="ner",
                )
            )
        return result

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error


# 3. LLM Semantic Extraction - report section 3.3
class LLMSemanticExtractor:
    """
    Calls an OpenAI-compatible chat endpoint -- local llama.cpp/llama-server
    serving Gemma 3 by default (see config.py LLMConfig), also works with
    Together AI, OpenRouter, or any other OpenAI-compatible host if you
    override EXTRACTION_LLM_BASE_URL/EXTRACTION_LLM_MODEL -- to extract
    request_type / topic / intent / keywords.

    Retries once with a stricter prompt if the first response fails to
    parse or comes back below the confidence threshold (report section 10).
    """

    def __init__(self, cfg: ExtractionAgentConfig = DEFAULT_CONFIG):
        self.cfg = cfg.llm
        self.threshold = cfg.confidence_threshold
        self.max_retries = cfg.max_llm_retries
        self._client = None
        self._init_error: Optional[str] = None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if not self.cfg.enabled:
            self._init_error = "LLM devre disi (config.llm.enabled=False)"
            return False
        # Local llama.cpp/llama-server (Gemma 3 default, see config.py) does
        # not require a real API key -- only external cloud hosts (e.g. the
        # old Together AI setup) do. Previously this hard-required a key
        # and silently skipped the whole LLM step if unset, which would
        # have broken extraction entirely against a keyless local server.
        # Matches VisionFieldExtractor's existing "not-needed" fallback
        # pattern in this same file.
        api_key = os.getenv(self.cfg.api_key_env) or "not-needed"
        try:
            from openai import OpenAI  # local import: optional dependency

            self._client = OpenAI(api_key=api_key, base_url=self.cfg.base_url, timeout=self.cfg.timeout_s)
            return True
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"LLM istemcisi baslatilamadi: {exc}"
            logger.warning(self._init_error)
            return False

    def _call(self, prompt: str) -> Optional[str]:
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM cagrisi basarisiz: %s", exc)
            return None

    def extract(self, text: str, classification_hint: Optional[str] = None) -> dict[str, Any]:
        """
        Returns a dict:
            {
              "data": {...parsed schema or {}...},
              "used": bool,
              "retried": bool,
              "retry_count": int,
              "error": str | None,
            }
        """
        out = {"data": {}, "used": False, "retried": False, "retry_count": 0, "error": None}
        if not text:
            out["error"] = "Bos metin"
            return out
        if not self._ensure_client():
            out["error"] = self._init_error
            return out

        attempt = 0
        retry = False
        while attempt <= self.max_retries:
            prompt = build_extraction_prompt(text, classification_hint, retry=retry)
            raw = self._call(prompt)
            out["used"] = True
            parsed = safe_json_loads(raw) if raw else None
            confidence = float(parsed.get("confidence", 0.0)) if parsed else 0.0

            if parsed is not None and confidence >= self.threshold:
                out["data"] = parsed
                out["retried"] = attempt > 0
                out["retry_count"] = attempt
                return out

            if parsed is not None and attempt == self.max_retries:
                # last attempt, even if low confidence - better than nothing,
                # Validation Agent will flag it via meta.low_confidence.
                out["data"] = parsed
                out["retried"] = attempt > 0
                out["retry_count"] = attempt
                return out

            attempt += 1
            retry = True

        out["error"] = "LLM gecerli/guvenilir JSON uretemedi (retry sonrasi da)"
        return out


# 3b. LangExtract grounded entity extraction - persons/organizations ONLY.
#
# Why only these two fields (not request_type/topic/intent too): LangExtract's
# value is character-span alignment against the *original* text, which only
# works for spans that appear verbatim (names, org names). Classification-
# style fields (request_type/topic/intent) are the model's own paraphrase/
# label, not a literal substring, so the aligner cannot ground them and
# there is no benefit over the plain LLMSemanticExtractor path already
# handling those fields. This was verified empirically before wiring it in
# here: verbatim spans ("Ahmet Yilmaz", "Enerji Mudurlugu") align correctly
# with char_interval; label-style text ("Sikayet") does not and comes back
# with char_interval=None, i.e. LangExtract would be doing nothing useful
# for those fields except adding a second LLM call.
_LANGEXTRACT_EXAMPLES: list[Any] | None = None


def _build_langextract_examples() -> list[Any]:
    global _LANGEXTRACT_EXAMPLES
    if _LANGEXTRACT_EXAMPLES is not None:
        return _LANGEXTRACT_EXAMPLES
    import langextract as lx

    _LANGEXTRACT_EXAMPLES = [
        lx.data.ExampleData(
            text=(
                "Elektrik faturam beklediğimden yüksek geldi. İncelenmesini "
                "istiyorum. Ahmet Yılmaz, Enerji Müdürlüğü'ne başvurmuştur."
            ),
            extractions=[
                lx.data.Extraction(extraction_class="persons", extraction_text="Ahmet Yılmaz"),
                lx.data.Extraction(extraction_class="organizations", extraction_text="Enerji Müdürlüğü"),
            ],
        ),
        lx.data.ExampleData(
            text="Dilekçe Sayın Vali Yardımcısı Mehmet Demir tarafından İmar Müdürlüğü'ne iletilmiştir.",
            extractions=[
                lx.data.Extraction(extraction_class="persons", extraction_text="Mehmet Demir"),
                lx.data.Extraction(extraction_class="organizations", extraction_text="İmar Müdürlüğü"),
            ],
        ),
    ]
    return _LANGEXTRACT_EXAMPLES


def _span_from_char_interval(extraction: Any) -> Optional[dict[str, int]]:
    interval = getattr(extraction, "char_interval", None)
    if interval is None or interval.start_pos is None or interval.end_pos is None:
        return None
    return {"start": int(interval.start_pos), "end": int(interval.end_pos)}


class LangExtractSemanticExtractor:
    """Grounded persons/organizations extraction via Google LangExtract,
    reusing the SAME OpenAI-compatible endpoint already configured for
    LLMSemanticExtractor (cfg.llm.base_url/model) -- no separate model or
    infra required. Fault-tolerant like every other tool in this file:
    any missing package / init / call failure degrades to an empty,
    clearly-flagged result rather than raising (see module docstring).
    """

    def __init__(self, cfg: ExtractionAgentConfig = DEFAULT_CONFIG):
        self.cfg = cfg.llm
        self.passes = cfg.llm.langextract_extraction_passes
        self.max_char_buffer = cfg.llm.langextract_max_char_buffer

    def extract_entities(self, text: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "persons": [],
            "organizations": [],
            "persons_spans": [],
            "organizations_spans": [],
            "used": False,
            "error": None,
        }
        if not text:
            return out

        try:
            import langextract as lx
            from langextract.data import FormatType
            from langextract.providers.openai import OpenAILanguageModel
        except ImportError as exc:
            out["error"] = f"langextract paketi kurulu degil, atlaniyor: {exc}"
            return out

        api_key = os.getenv(self.cfg.api_key_env) or "not-needed"
        try:
            model = OpenAILanguageModel(
                model_id=self.cfg.model,
                api_key=api_key,
                base_url=self.cfg.base_url,
                temperature=0.0,
            )
            result = lx.extract(
                text_or_documents=text,
                prompt_description=(
                    "Turkce resmi bir belge metninden SADECE metinde birebir "
                    "gecen gercek kisi ad-soyadlarini (persons) ve kurum/"
                    "mudurluk isimlerini (organizations) cikar. Metinde "
                    "olmayan hicbir sey uydurma; unvan/rolleri ('Sayin "
                    "Yetkili' gibi) kisi adi olarak sayma."
                ),
                examples=_build_langextract_examples(),
                model=model,
                format_type=FormatType.JSON,
                fence_output=False,
                use_schema_constraints=False,
                extraction_passes=max(1, self.passes),
                max_char_buffer=self.max_char_buffer,
                show_progress=False,
            )
        except Exception as exc:  # noqa: BLE001 - must never crash the agent
            logger.warning("LangExtract cagrisi basarisiz: %s", exc)
            out["error"] = f"LangExtract cagrisi basarisiz: {exc}"
            return out

        out["used"] = True
        for extraction in result.extractions or []:
            value = (extraction.extraction_text or "").strip()
            if not value:
                continue
            span = _span_from_char_interval(extraction)
            if extraction.extraction_class == "persons":
                out["persons"].append(value)
                out["persons_spans"].append(span)
            elif extraction.extraction_class == "organizations":
                out["organizations"].append(value)
                out["organizations_spans"].append(span)
        return out


class HybridSemanticExtractor:
    """Drop-in replacement for LLMSemanticExtractor with the exact same
    `.extract(text, classification_hint) -> dict` contract, so agent.py
    needs no changes beyond swapping which class it instantiates.

    Behavior: runs the existing plain LLM call for request_type/topic/
    intent/keywords/missing_info/confidence (unchanged), then -- only if
    cfg.llm.use_langextract is True -- additionally grounds persons/
    organizations via LangExtractSemanticExtractor and overrides those two
    fields with the grounded, span-verified values when available. Any
    LangExtract failure silently keeps the plain call's persons/
    organizations output (which is exactly what ran before this change),
    so this can never make extraction_agent worse than it was.
    """

    def __init__(self, cfg: ExtractionAgentConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self._plain = LLMSemanticExtractor(cfg)
        self._grounded = LangExtractSemanticExtractor(cfg) if cfg.llm.use_langextract else None

    def extract(self, text: str, classification_hint: Optional[str] = None) -> dict[str, Any]:
        out = self._plain.extract(text, classification_hint=classification_hint)
        if not self._grounded:
            out["langextract_used"] = False
            return out

        grounded = self._grounded.extract_entities(text)
        out["langextract_used"] = bool(grounded["used"] and not grounded["error"])
        if grounded["error"]:
            out["langextract_error"] = grounded["error"]

        if out["langextract_used"]:
            data = out.setdefault("data", {})
            if grounded["persons"]:
                data["persons"] = grounded["persons"]
                data["persons_spans"] = grounded["persons_spans"]
            if grounded["organizations"]:
                data["organizations"] = grounded["organizations"]
                data["organizations_spans"] = grounded["organizations_spans"]
        return out


# 4. Vision Extraction (Qwen-VL) - report section 7
class VisionFieldExtractor:
    

    def __init__(self, cfg: ExtractionAgentConfig = DEFAULT_CONFIG):
        self.cfg = cfg.vision
        self._client = None
        self._init_error: Optional[str] = None

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        if not self.cfg.enabled or not self.cfg.base_url:
            self._init_error = "Vision devre disi ya da base_url tanimsiz"
            return False
        api_key = os.getenv(self.cfg.api_key_env, "")
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key or "not-needed", base_url=self.cfg.base_url, timeout=self.cfg.timeout_s)
            return True
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"Vision istemcisi baslatilamadi: {exc}"
            logger.warning(self._init_error)
            return False

    def extract_from_image(self, image_b64: str) -> dict[str, Any]:
        out = {"data": {}, "used": False, "error": None}
        if not self._ensure_client():
            out["error"] = self._init_error
            return out
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_USER_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ],
                    },
                ],
                max_tokens=500,
            )
            raw = resp.choices[0].message.content
            parsed = safe_json_loads(raw)
            out["used"] = True
            out["data"] = parsed or {}
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vision cagrisi basarisiz: %s", exc)
            out["error"] = str(exc)
            return out