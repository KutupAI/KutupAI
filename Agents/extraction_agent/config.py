"""
---------
Çıkarma aracısına özgü yapılandırma (eşikler, yardımcı model yolları).

Her şey ortam değişkenleri aracılığıyla geçersiz kılınabilir, böylece aynı kod
barındırılan bir API'ye (Together / OpenRouter / herhangi bir OpenAI uyumlu Qwen
uç noktası) veya yerel olarak sunulan bir modele (vLLM / Ollama / TGI de OpenAI uyumlu /v1 uç noktası sunar) karşı koda dokunmadan çalışır - sadece
.env dosyasını değiştirin.

Gizli bilgileri (API anahtarları) kaynak kontrolünden uzak tutun - bunları yalnızca ortam değişkenlerinden okuyun.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# .env dosyasini burada, en erken noktada yukluyoruz. Neden burada?
# `python -m Agents.extraction_agent.test_agent` calistirildiginda, Python
# ONCE Agents/extraction_agent/__init__.py'yi calistirir (bu da agent.py ->
# config.py zincirini tetikler), SONRA test_agent.py'yi __main__ olarak
# calistirir. Yani test_agent.py icindeki load_dotenv() cok GEC kalirdi -
# config.py'nin asagidaki dataclass alanlari zaten (yanlis/varsayilan)
# degerlerle tanimlanmis olurdu. Bu yuzden .env yuklemesi, env okuyan ilk
# modul olan burada (config.py) yapilmali.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  


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


@dataclass
class RegexConfig:
    """Rule-based extraction patterns (report section 3.1 / 4)."""

    # Turkish date formats: 12.05.2024 | 12/05/2024 | 12 Mayis 2024 | 2024-05-12
    date_patterns: List[str] = field(
        default_factory=lambda: [
            r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b",
            r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
            r"\b(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|"
            r"Eylül|Ekim|Kasım|Aralık)\s+(\d{4})\b",
        ]
    )
    # Turkish phone numbers: 0532 123 45 67 | +90 532 123 4567 | (0212) 123 45 67
    phone_patterns: List[str] = field(
        default_factory=lambda: [
            r"(?:\+90|0)?\s*\(?5\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}",
            r"(?:\+90|0)?\s*\(?0?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}",
        ]
    )
    email_pattern: str = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    # Evrak/sayi numarasi: "Sayı: 2024/1234", "Evrak No: E-12345", "No: 456789"
    evrak_no_patterns: List[str] = field(
        default_factory=lambda: [
            r"(?:Sayı|Sayi|Evrak\s*No|Belge\s*No)\s*[:\-]?\s*([A-Za-z0-9\-/]{3,20})",
            r"\bE-\d{6,}\b",
        ]
    )


@dataclass
class NERConfig:
    """
    NLP entity extraction (report section 3.2).

    Devre disi birakildi (varsayilan): Kisi/kurum adlari artik ayni LLM
    cagrisi (Qwen-VL) icinden "persons"/"organizations" alanlariyla
    aliniyor - bu, ekstra transformers/torch kurulumunu ve model indirmeyi
    gerektirmiyor. Ayri, ozel bir NER modeli isterseniz
    EXTRACTION_NER_ENABLED=true yaparak tekrar aktif edebilirsiniz.
    """

    enabled: bool = _env_bool("EXTRACTION_NER_ENABLED", False)
    # Turkish NER models trained for PERSON/ORG/LOC - swap freely via env.
    model_name: str = os.getenv("EXTRACTION_NER_MODEL", "akdeniz27/bert-base-turkish-cased-ner")
    device: str = os.getenv("EXTRACTION_NER_DEVICE", "cpu")  # "cpu" | "cuda" | "cuda:0"
    aggregation_strategy: str = "simple"
    max_chars: int = _env_int("EXTRACTION_NER_MAX_CHARS", 4000)  # avoid OOM on huge docs


@dataclass
class LLMConfig:
    """LLM semantic extraction (report section 3.3 / 7)."""

    enabled: bool = _env_bool("EXTRACTION_LLM_ENABLED", True)
    base_url: str = os.getenv("EXTRACTION_LLM_BASE_URL", "https://api.together.xyz/v1")
    api_key_env: str = "EXTRACTION_LLM_API_KEY"
    model: str = os.getenv("EXTRACTION_LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct-Turbo")
    temperature: float = _env_float("EXTRACTION_LLM_TEMPERATURE", 0.1)
    max_tokens: int = _env_int("EXTRACTION_LLM_MAX_TOKENS", 800)
    timeout_s: int = _env_int("EXTRACTION_LLM_TIMEOUT", 30)


@dataclass
class VisionConfig:
    """Qwen-VL vision extraction (report section 7) - imza/kase/tablo/el yazisi."""

    enabled: bool = _env_bool("EXTRACTION_VLM_ENABLED", True)
    base_url: str = os.getenv("EXTRACTION_VLM_BASE_URL", "")
    api_key_env: str = "EXTRACTION_VLM_API_KEY"
    model: str = os.getenv("EXTRACTION_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    timeout_s: int = _env_int("EXTRACTION_VLM_TIMEOUT", 45)


@dataclass
class ExtractionAgentConfig:
    """Top-level config bundle used by agent.py / tools.py."""

    confidence_threshold: float = _env_float("EXTRACTION_CONF_THRESHOLD", 0.55)
    max_llm_retries: int = _env_int("EXTRACTION_LLM_MAX_RETRIES", 1)
    regex: RegexConfig = field(default_factory=RegexConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)


DEFAULT_CONFIG = ExtractionAgentConfig()