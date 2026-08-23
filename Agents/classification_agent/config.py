"""
config.py
-----------
Configuration specific to classification_agent (thresholds, model paths).

IMPORTANT -- needs_review_threshold default:
Section 7 of the task document is explicit that this value must be determined
experimentally on the validation set, not picked arbitrarily ("kafadan
%50/%70 gibi bir esik secilmemeli"). The 0.60 default below is a
placeholder ONLY so the agent is runnable before real labeled data exists.
It MUST be re-tuned (see Optimization/models/model_metadata.json /
evaluation scripts once section 8 metrics are available) before this is
treated as a final value in the deliverable.

MODEL SWITCH (Qwen2.5-VL -> Gemma 3): field names below were renamed from
qwen_* to vlm_* since the classification VLM is no longer Qwen-specific
(Gemma 3 4B/12B/27B, served locally via llama.cpp/llama-server, per the
project's Gemma3-everywhere policy). Old QWEN_VLM_* env vars are still
read as a fallback (see from_env()) so existing .env files keep working
until updated; the new VLM_* names take precedence when both are set.
qwen_* attribute names are kept as read-only aliases below for any caller
not yet updated (e.g. evaluation/ablation.py comments reference them).
"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ClassificationConfig:

    needs_review_threshold: float = 0.60

    top_k_alternatives: int = 2

    use_fast_classifier: bool = True
    fast_classifier_escalation_threshold: float = 0.75

    # --- VLM (multimodal fallback / primary per section 4) ---
    # Gemma 3 (4B/12B/27B) served locally via llama.cpp/llama-server.
    # llama-server ignores `model` for a single loaded model, but it is
    # still sent (see Inference/client/vlm_client.py) for forward-compat
    # with multi-model routing setups. IMPORTANT: llama-server must be
    # started with BOTH --model <gemma3 gguf> AND --mmproj <its vision
    # projector gguf>, or image input is silently ignored.
    vlm_base_url: str = "http://localhost:8092/v1/chat/completions"
    vlm_model_name: str = "gemma-3-27b-it"
    vlm_timeout_s: int = 300
    vlm_temperature: float = 0.0
    vlm_max_tokens: int = 512

    send_image: bool = True
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
        def boolean(name: str, default: bool) -> bool:
            value = os.getenv(name)
            return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

        def env_with_legacy_fallback(new_name: str, legacy_name: str, default: str) -> str:
            # New VLM_* var wins; fall back to the old QWEN_VLM_* var (still
            # honored) so existing .env files don't break on upgrade; only
            # if neither is set do we use `default`.
            return os.getenv(new_name, os.getenv(legacy_name, default))

        return cls(
            needs_review_threshold=float(os.getenv("CLASSIFICATION_NEEDS_REVIEW_THRESHOLD", "0.60")),
            top_k_alternatives=int(os.getenv("CLASSIFICATION_TOP_K_ALTERNATIVES", "2")),
            use_fast_classifier=boolean("CLASSIFICATION_USE_FAST_CLASSIFIER", True),
            fast_classifier_escalation_threshold=float(
                os.getenv("CLASSIFICATION_FAST_ESCALATION_THRESHOLD", "0.75")
            ),
            vlm_base_url=env_with_legacy_fallback(
                "VLM_BASE_URL", "QWEN_VLM_BASE_URL", "http://localhost:8092/v1/chat/completions"
            ),
            vlm_model_name=env_with_legacy_fallback("VLM_MODEL_NAME", "QWEN_VLM_MODEL_NAME", "gemma-3-27b-it"),
            vlm_timeout_s=int(env_with_legacy_fallback("VLM_TIMEOUT_S", "QWEN_VLM_TIMEOUT_S", "300")),
            vlm_temperature=float(env_with_legacy_fallback("VLM_TEMPERATURE", "QWEN_VLM_TEMPERATURE", "0.0")),
            vlm_max_tokens=int(env_with_legacy_fallback("VLM_MAX_TOKENS", "QWEN_VLM_MAX_TOKENS", "512")),
            send_image=boolean("CLASSIFICATION_SEND_IMAGE", True),
            max_image_dimension=int(os.getenv("CLASSIFICATION_MAX_IMAGE_DIMENSION", "1600")),
            use_layout_when_available=boolean("CLASSIFICATION_USE_LAYOUT", True),
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