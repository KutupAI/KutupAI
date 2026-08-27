"""Summary agent config (env + model registry)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else float(value)


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else int(value)


def _str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


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


@dataclass(frozen=True)
class SummaryConfig:
    max_tokens: int = 700
    temperature: float = 0.2
    no_context_marker: str = "no_relevant_context"
    inference_url: str = "http://127.0.0.1:8082/v1/chat/completions"
    timeout: int = 300
    model_name: str = "gemma3"
    inference_backend: str = "evren"
    evren_model: str = "llm-large"

    @classmethod
    def from_env(cls) -> SummaryConfig:
        registry = _load_model_registry()
        host = _str("INFERENCE_HOST", "127.0.0.1")
        port = _int("INFERENCE_PORT", 8082)
        default_url = f"http://{host}:{port}/v1/chat/completions"

        return cls(
            max_tokens=_int("SUMMARY_MAX_TOKENS", 700),
            temperature=_float("SUMMARY_TEMPERATURE", 0.2),
            no_context_marker=_str("SUMMARY_NO_CONTEXT_MARKER", "no_relevant_context") or "no_relevant_context",
            inference_url=_str("INFERENCE_URL", default_url) or default_url,
            timeout=_int("INFERENCE_TIMEOUT", 300),
            model_name=_str("SUMMARY_MODEL_NAME", registry.get("model_name", "gemma3")) or "gemma3",
            inference_backend=(_str("SUMMARY_INFERENCE_BACKEND", "evren") or "evren").casefold(),
            evren_model=_str("SUMMARY_EVREN_MODEL", "llm-large") or "llm-large",
        )
