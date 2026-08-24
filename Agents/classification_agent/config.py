"""
config.py
-----------
Configuration specific to classification_agent (thresholds, generation).

Inference ownership:
  Model name, GGUF path, and server host/port come from the Inference layer
  (Inference/models/model_registry.json + Inference/configuration/
  inference_config.yaml), same pattern as summary_agent / writer_agent.
  This file only holds classification-specific knobs (thresholds, image
  toggle, token limits).

IMPORTANT -- needs_review_threshold default:
Section 7 of the task document is explicit that this value must be determined
experimentally on the validation set, not picked arbitrarily. The 0.60
default below is a placeholder ONLY so the agent is runnable before real
labeled data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


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


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _default_inference_url() -> str:
    server = _load_inference_server()
    host = os.getenv("INFERENCE_HOST") or server.get("host") or "127.0.0.1"
    port = int(os.getenv("INFERENCE_PORT") or server.get("port") or 8080)
    return f"http://{host}:{port}/v1/chat/completions"


@dataclass(frozen=True)
class ClassificationConfig:

    needs_review_threshold: float = 0.60

    top_k_alternatives: int = 2

    use_fast_classifier: bool = True
    fast_classifier_escalation_threshold: float = 0.75

    # --- Inference (shared Gemma3 via Inference/llama_server on :8080) ---
    # Defaults match Inference/models/model_registry.json + inference_config.yaml.
    # Override with INFERENCE_URL / INFERENCE_HOST / INFERENCE_PORT / VLM_*.
    vlm_base_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    vlm_model_name: str = "gemma3"
    vlm_timeout_s: int = 300
    vlm_temperature: float = 0.0
    vlm_max_tokens: int = 512

    # Shared Inference launcher loads gemma3.gguf without --mmproj, so images
    # are ignored unless a multimodal projector is added. Default off.
    send_image: bool = False
    max_image_dimension: int = 1600

    # --- inputs ---
    use_layout_when_available: bool = True

    # --- backward-compat read-only aliases (pre-Gemma-migration naming) ---
    @property
    def qwen_base_url(self) -> str:
        return self.vlm_base_url

    @property
    def qwen_model_name(self) -> str:
        return self.vlm_model_name

    @property
    def qwen_timeout_s(self) -> int:
        return self.vlm_timeout_s

    @property
    def qwen_temperature(self) -> float:
        return self.vlm_temperature

    @property
    def qwen_max_tokens(self) -> int:
        return self.vlm_max_tokens

    @classmethod
    def from_env(cls) -> "ClassificationConfig":
        registry = _load_model_registry()
        default_url = os.getenv("INFERENCE_URL") or _default_inference_url()
        default_model = registry.get("model_name") or "gemma3"
        default_timeout = int(os.getenv("INFERENCE_TIMEOUT", "300"))

        return cls(
            needs_review_threshold=float(os.getenv("CLASSIFICATION_NEEDS_REVIEW_THRESHOLD", "0.60")),
            top_k_alternatives=int(os.getenv("CLASSIFICATION_TOP_K_ALTERNATIVES", "2")),
            use_fast_classifier=_boolean("CLASSIFICATION_USE_FAST_CLASSIFIER", True),
            fast_classifier_escalation_threshold=float(
                os.getenv("CLASSIFICATION_FAST_ESCALATION_THRESHOLD", "0.75")
            ),
            vlm_base_url=_env_str("VLM_BASE_URL", "QWEN_VLM_BASE_URL", default=default_url),
            vlm_model_name=_env_str(
                "VLM_MODEL_NAME", "QWEN_VLM_MODEL_NAME", "CLASSIFICATION_MODEL_NAME",
                default=str(default_model),
            ),
            vlm_timeout_s=int(
                _env_str("VLM_TIMEOUT_S", "QWEN_VLM_TIMEOUT_S", default=str(default_timeout))
            ),
            vlm_temperature=float(
                _env_str("VLM_TEMPERATURE", "QWEN_VLM_TEMPERATURE", default="0.0")
            ),
            vlm_max_tokens=int(
                _env_str("VLM_MAX_TOKENS", "QWEN_VLM_MAX_TOKENS", default="512")
            ),
            send_image=_boolean("CLASSIFICATION_SEND_IMAGE", False),
            max_image_dimension=int(os.getenv("CLASSIFICATION_MAX_IMAGE_DIMENSION", "1600")),
            use_layout_when_available=_boolean("CLASSIFICATION_USE_LAYOUT", True),
        )

    def validate(self) -> None:
        if not 0 <= self.needs_review_threshold <= 1:
            raise ValueError("needs_review_threshold must be between 0 and 1.")
        if not 0 <= self.fast_classifier_escalation_threshold <= 1:
            raise ValueError("fast_classifier_escalation_threshold must be between 0 and 1.")
        if self.top_k_alternatives < 0:
            raise ValueError("top_k_alternatives must be >= 0.")
        if self.vlm_timeout_s <= 0:
            raise ValueError("vlm_timeout_s must be positive.")
