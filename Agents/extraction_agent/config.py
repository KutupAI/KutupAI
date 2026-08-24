"""
config.py
-----------
Configuration specific to extraction_agent (thresholds, helpers).

Inference ownership:
  Model name, GGUF path, and server host/port come from the Inference layer
  (Inference/models/model_registry.json + Inference/configuration/
  inference_config.yaml), same pattern as classification_agent / summary_agent.
  This file only holds extraction-specific knobs (regex/NER/LLM/vision toggles).

Secrets (API keys) stay out of source control -- read from env only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_model_registry() -> dict:
    registry_path = _project_root() / "Inference" / "models" / "model_registry.json"
    if not registry_path.is_file():
        return {}
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_inference_server() -> dict:
    """Read host/port from Inference/configuration/inference_config.yaml."""
    config_path = _project_root() / "Inference" / "configuration" / "inference_config.yaml"
    if not config_path.is_file():
        return {}
    host, port = "127.0.0.1", 8080
    try:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("host:"):
                host = stripped.split(":", 1)[1].strip().strip("\"'")
            elif stripped.startswith("port:"):
                port = int(stripped.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return {}
    return {"host": host, "port": port}


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_str(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _default_openai_base_url() -> str:
    """OpenAI-compatible base (…/v1), not the full chat/completions URL."""
    server = _load_inference_server()
    host = os.getenv("INFERENCE_HOST") or server.get("host") or "127.0.0.1"
    port = int(os.getenv("INFERENCE_PORT") or server.get("port") or 8080)
    return f"http://{host}:{port}/v1"


def _default_model_name() -> str:
    registry = _load_model_registry()
    return str(registry.get("model_name") or "gemma3")


@dataclass
class RegexConfig:
    """Rule-based extraction patterns (report section 3.1 / 4)."""

    date_patterns: List[str] = field(
        default_factory=lambda: [
            r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b",
            r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
            r"\b(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|"
            r"Eylül|Ekim|Kasım|Aralık)\s+(\d{4})\b",
        ]
    )
    phone_patterns: List[str] = field(
        default_factory=lambda: [
            r"(?:\+90|0)?\s*\(?5\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}",
            r"(?:\+90|0)?\s*\(?0?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}",
        ]
    )
    email_pattern: str = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    evrak_no_patterns: List[str] = field(
        default_factory=lambda: [
            r"(?:Sayı|Sayi|Evrak\s*No|Belge\s*No)\s*[:\-]?\s*([A-Za-z0-9\-/]{3,20})",
            r"\bE-\d{6,}\b",
        ]
    )


@dataclass
class NERConfig:
    """NLP entity extraction (report section 3.2). Disabled by default."""

    enabled: bool = _env_bool("EXTRACTION_NER_ENABLED", False)
    model_name: str = os.getenv("EXTRACTION_NER_MODEL", "akdeniz27/bert-base-turkish-cased-ner")
    device: str = os.getenv("EXTRACTION_NER_DEVICE", "cpu")
    aggregation_strategy: str = "simple"
    max_chars: int = _env_int("EXTRACTION_NER_MAX_CHARS", 4000)


@dataclass
class LLMConfig:
    """LLM semantic extraction via Inference llama-server (Gemma 3).

    Defaults match Inference/models/model_registry.json + inference_config.yaml.
    Override with EXTRACTION_LLM_* or shared INFERENCE_* env vars.
    """

    enabled: bool = _env_bool("EXTRACTION_LLM_ENABLED", True)
    base_url: str = field(default_factory=_default_openai_base_url)
    api_key_env: str = "EXTRACTION_LLM_API_KEY"
    model: str = field(default_factory=_default_model_name)
    temperature: float = _env_float("EXTRACTION_LLM_TEMPERATURE", 0.1)
    max_tokens: int = _env_int("EXTRACTION_LLM_MAX_TOKENS", 800)
    timeout_s: int = _env_int("EXTRACTION_LLM_TIMEOUT", 300)

    use_langextract: bool = _env_bool("EXTRACTION_USE_LANGEXTRACT", True)
    langextract_extraction_passes: int = _env_int("EXTRACTION_LANGEXTRACT_PASSES", 1)
    langextract_max_char_buffer: int = _env_int("EXTRACTION_LANGEXTRACT_MAX_CHAR_BUFFER", 4000)

    # --- LangExtract (grounded/schema-based extraction) ---
    # When True, LLMSemanticExtractor's role is taken over by
    # LangExtractSemanticExtractor (character-span-grounded persons/
    # organizations/topic/etc, same OpenAI-compatible endpoint as above --
    # no separate model or infra). Falls back automatically to the plain
    # prompt-based extractor (use_langextract path off) if the langextract
    # package is missing or a call fails, per this file's fault-tolerance
    # rule -- never a hard dependency.
    use_langextract: bool = _env_bool("EXTRACTION_USE_LANGEXTRACT", True)
    langextract_extraction_passes: int = _env_int("EXTRACTION_LANGEXTRACT_PASSES", 1)
    langextract_max_char_buffer: int = _env_int("EXTRACTION_LANGEXTRACT_MAX_CHAR_BUFFER", 4000)


@dataclass
class VisionConfig:
    """Vision extraction via the same Inference server when enabled."""

    enabled: bool = _env_bool("EXTRACTION_VLM_ENABLED", True)
    base_url: str = field(default_factory=_default_openai_base_url)
    api_key_env: str = "EXTRACTION_VLM_API_KEY"
    model: str = field(default_factory=_default_model_name)
    timeout_s: int = _env_int("EXTRACTION_VLM_TIMEOUT", 300)


@dataclass
class ExtractionAgentConfig:
    """Top-level config bundle used by agent.py / tools.py."""

    confidence_threshold: float = _env_float("EXTRACTION_CONF_THRESHOLD", 0.55)
    max_llm_retries: int = _env_int("EXTRACTION_LLM_MAX_RETRIES", 1)
    regex: RegexConfig = field(default_factory=RegexConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    @classmethod
    def from_env(cls) -> "ExtractionAgentConfig":
        """Build config with Inference defaults, then EXTRACTION_* overrides."""
        default_url = _env_str(
            "EXTRACTION_LLM_BASE_URL",
            "INFERENCE_OPENAI_BASE_URL",
            default=_default_openai_base_url(),
        )
        # Allow full chat URL; OpenAI client wants …/v1
        if default_url.rstrip("/").endswith("/chat/completions"):
            default_url = default_url.rstrip("/").removesuffix("/chat/completions")
        default_model = _env_str(
            "EXTRACTION_LLM_MODEL",
            "INFERENCE_MODEL_NAME",
            default=_default_model_name(),
        )
        default_timeout = _env_int("EXTRACTION_LLM_TIMEOUT", _env_int("INFERENCE_TIMEOUT", 300))
        vision_url = _env_str(
            "EXTRACTION_VLM_BASE_URL",
            default=default_url,
        )
        if vision_url.rstrip("/").endswith("/chat/completions"):
            vision_url = vision_url.rstrip("/").removesuffix("/chat/completions")
        vision_model = _env_str("EXTRACTION_VLM_MODEL", default=default_model)

        return cls(
            confidence_threshold=_env_float("EXTRACTION_CONF_THRESHOLD", 0.55),
            max_llm_retries=_env_int("EXTRACTION_LLM_MAX_RETRIES", 1),
            ner=NERConfig(),
            llm=LLMConfig(
                enabled=_env_bool("EXTRACTION_LLM_ENABLED", True),
                base_url=default_url,
                model=default_model,
                temperature=_env_float("EXTRACTION_LLM_TEMPERATURE", 0.1),
                max_tokens=_env_int("EXTRACTION_LLM_MAX_TOKENS", 800),
                timeout_s=default_timeout,
                use_langextract=_env_bool("EXTRACTION_USE_LANGEXTRACT", True),
                langextract_extraction_passes=_env_int("EXTRACTION_LANGEXTRACT_PASSES", 1),
                langextract_max_char_buffer=_env_int(
                    "EXTRACTION_LANGEXTRACT_MAX_CHAR_BUFFER", 4000
                ),
            ),
            vision=VisionConfig(
                enabled=_env_bool("EXTRACTION_VLM_ENABLED", True),
                base_url=vision_url or default_url,
                model=vision_model,
                timeout_s=_env_int("EXTRACTION_VLM_TIMEOUT", default_timeout),
            ),
        )


DEFAULT_CONFIG = ExtractionAgentConfig.from_env()
