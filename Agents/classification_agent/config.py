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

    # --- Qwen VLM (multimodal fallback / primary per section 4) ---
    qwen_base_url: str = "http://localhost:8092/v1/chat/completions"
    qwen_model_name: str = "qwen-vl"
    qwen_timeout_s: int = 300
    qwen_temperature: float = 0.0  
    qwen_max_tokens: int = 512
    
    send_image: bool = True
    max_image_dimension: int = 1600

    # --- inputs ---
    use_layout_when_available: bool = True

    @classmethod
    def from_env(cls) -> "ClassificationConfig":
        def boolean(name: str, default: bool) -> bool:
            value = os.getenv(name)
            return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            needs_review_threshold=float(os.getenv("CLASSIFICATION_NEEDS_REVIEW_THRESHOLD", "0.60")),
            top_k_alternatives=int(os.getenv("CLASSIFICATION_TOP_K_ALTERNATIVES", "2")),
            use_fast_classifier=boolean("CLASSIFICATION_USE_FAST_CLASSIFIER", True),
            fast_classifier_escalation_threshold=float(
                os.getenv("CLASSIFICATION_FAST_ESCALATION_THRESHOLD", "0.75")
            ),
            qwen_base_url=os.getenv("QWEN_VLM_BASE_URL", "http://localhost:8092/v1/chat/completions"),
            qwen_model_name=os.getenv("QWEN_VLM_MODEL_NAME", "qwen-vl"),
            qwen_timeout_s=int(os.getenv("QWEN_VLM_TIMEOUT_S", "300")),
            qwen_temperature=float(os.getenv("QWEN_VLM_TEMPERATURE", "0.0")),
            qwen_max_tokens=int(os.getenv("QWEN_VLM_MAX_TOKENS", "512")),
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
        if self.qwen_timeout_s <= 0:
            raise ValueError("qwen_timeout_s must be positive.")
