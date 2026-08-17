"""Vision fallback: last resort when OCR + retries still can't read a page.

Kept behind an interface so the concrete model (Qwen-VL today) can be
swapped later without touching the OCR pipeline.

IMPORTANT: this is a fallback, never the primary OCR path. It must stay
lazy-loaded (no network/model handle is created until `read_page` is
actually called) and it must never be invoked for every page.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib import error, request

import numpy as np

from Agents.ocr_agent.config import VisionFallbackConfig
from Agents.ocr_agent.exceptions import VisionFallbackError

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")

_TEXT_AND_VISION_PROMPT = (
    "Transcribe ALL visible printed text on this page, from the first line "
    "to the last line including headers, margins and the footer.\n"
    "Rules:\n"
    "- Copy every line. Do not skip, summarize, translate, or rewrite.\n"
    "- Preserve Turkish letters exactly: İ ı Ş ş Ğ ğ Ç ç Ö ö Ü ü.\n"
    "- Keep original paragraphs and line breaks. Do not mix lines from "
    "different paragraphs. Do not merge unrelated text blocks.\n"
    "- Reading order: top-to-bottom, left-to-right (left column fully, then right).\n"
    "- If a word is overlayed by ink, reconstruct the printed word only if it "
    "is still readable; never invent tokens.\n"
    "- Signature/stamp: true ONLY if you SEE handwritten ink or a physical "
    "stamp/seal. Printed words such as İmza/signature are NOT enough. "
    "CamScanner watermarks are NOT stamps.\n"
    "Return JSON only:\n"
    '{"text":"<full page text>","signature":{"detected":false,"handwritten":false},"stamp":{"detected":false}}'
)

_VISION_ONLY_PROMPT = (
    "Look at this document image. Detect only visible marks.\n"
    "signature.detected=true only if you SEE a handwritten or ink signature.\n"
    "handwritten=true only if that signature is handwritten.\n"
    "stamp.detected=true only if you SEE a physical stamp/seal/mühür.\n"
    "Do not use printed words like İmza, signature, mühür, stamp as proof.\n"
    "CamScanner watermarks are not stamps. Logos are not signatures.\n"
    "Return JSON only:\n"
    '{"signature":{"detected":false,"handwritten":false},"stamp":{"detected":false}}'
)

_QWEN_MAX_EDGE = 1792

_TR_FOLD = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "I": "i", "í": "i",
    "ö": "o", "ş": "s", "ü": "u",
    "Ç": "c", "Ğ": "g", "İ": "i", "Ö": "o", "Ş": "s", "Ü": "u",
})
_TR_LETTERS = set("çğıöşüÇĞİÖŞÜ")


@dataclass(frozen=True)
class VisionFallbackResult:
    text: str
    confidence: float | None
    uncertain: bool
    provider: str
    signature_detected: bool = False
    signature_handwritten: bool = False
    stamp_detected: bool = False


class VisionFallbackInterface(ABC):
    """Contract every vision-fallback provider must implement."""

    provider_name: str = "unknown"

    @abstractmethod
    def read_page(self, image: np.ndarray) -> VisionFallbackResult:
        """Best-effort transcription of a single page image."""
        raise NotImplementedError

    def inspect_visuals(self, image: np.ndarray) -> VisionFallbackResult:
        """Optional signature/stamp-only check. Default: same backend, no text."""
        return self.read_page(image)


class NullVisionFallback(VisionFallbackInterface):
    """Used when fallback is disabled/unconfigured — fails closed and loud."""

    provider_name = "disabled"

    def read_page(self, image: np.ndarray) -> VisionFallbackResult:
        raise VisionFallbackError(
            "Vision fallback was requested but is disabled/unconfigured "
            "(OCR_VISION_FALLBACK_ENABLED=false or no provider client available)."
        )

    def inspect_visuals(self, image: np.ndarray) -> VisionFallbackResult:
        raise VisionFallbackError("Vision fallback is disabled.")


class QwenVLVisionFallback(VisionFallbackInterface):
    """Qwen2.5-VL fallback against a local OpenAI-compatible llama-server."""

    provider_name = "qwen-vl"

    def __init__(self, config: VisionFallbackConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client  # lazy: only resolved on first real call

    def _ensure_http_ready(self) -> str:
        endpoint = (self.config.endpoint or "").strip()
        if not endpoint:
            raise VisionFallbackError("OCR_VISION_FALLBACK_ENDPOINT is not configured.")
        if not endpoint.startswith("http"):
            raise VisionFallbackError(f"Invalid Qwen-VL endpoint: {endpoint}")
        return endpoint

    def _maybe_shared_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        if not self.config.use_shared_inference_client:
            return None
        try:
            from Inference.client.llama_client import VisionInferenceClient  # type: ignore

            self._client = VisionInferenceClient()
            return self._client
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "Shared Inference vision client unavailable (%s); "
                "using local Qwen-VL HTTP endpoint.",
                exc,
            )
            return None

    def read_page(self, image: np.ndarray) -> VisionFallbackResult:
        return self._call(image, _TEXT_AND_VISION_PROMPT, max_tokens=self.config.max_tokens)

    def inspect_visuals(self, image: np.ndarray) -> VisionFallbackResult:
        return self._call(image, _VISION_ONLY_PROMPT, max_tokens=min(512, self.config.max_tokens))

    def _call(self, image: np.ndarray, prompt: str, *, max_tokens: int) -> VisionFallbackResult:
        shared = self._maybe_shared_client()
        try:
            if shared is not None and hasattr(shared, "generate_vision"):
                image_b64 = _encode_image_b64(image, mime="png")
                response = shared.generate_vision(
                    image_base64=image_b64,
                    instructions=prompt,
                    timeout_s=self.config.request_timeout_s,
                )
                raw_text = str(getattr(response, "text", response) or "").strip()
                confidence = getattr(response, "confidence", None)
            else:
                endpoint = self._ensure_http_ready()
                raw_text = _chat_completions(
                    endpoint=endpoint,
                    model=self.config.model_name,
                    prompt=prompt,
                    image_b64=_encode_image_b64(image, mime="jpeg"),
                    mime="image/jpeg",
                    timeout_s=self.config.request_timeout_s,
                    max_tokens=max_tokens,
                )
                confidence = None
        except VisionFallbackError:
            raise
        except Exception as exc:
            raise VisionFallbackError(f"Qwen-VL fallback call failed: {exc}") from exc

        parsed = _parse_vision_json(raw_text)
        text = _usable_page_text(str(parsed.get("text") or ""))
        signature = parsed.get("signature") if isinstance(parsed.get("signature"), dict) else {}
        stamp = parsed.get("stamp") if isinstance(parsed.get("stamp"), dict) else {}
        return VisionFallbackResult(
            text=text,
            confidence=confidence,
            uncertain=True,
            provider=self.provider_name,
            signature_detected=bool(signature.get("detected")),
            signature_handwritten=bool(signature.get("handwritten")),
            stamp_detected=bool(stamp.get("detected")),
        )


def build_vision_fallback(config: VisionFallbackConfig) -> VisionFallbackInterface:
    if not config.enabled:
        return NullVisionFallback()
    if config.provider == "qwen-vl":
        return QwenVLVisionFallback(config)
    logger.warning("Unknown vision fallback provider '%s'; disabling fallback.", config.provider)
    return NullVisionFallback()


def choose_better_text(ocr_text: str, qwen_text: str) -> tuple[str, bool]:
    """Keep OCR unless Qwen recovers missing/corrupted regions."""

    return merge_ocr_and_vision(ocr_text, qwen_text)


def merge_ocr_and_vision(
    ocr_text: str,
    qwen_text: str,
    *,
    incomplete: bool = False,
    corrupted: bool = False,
) -> tuple[str, bool]:
    """Use Qwen to fill gaps / verify spelling. Do not blindly replace good OCR."""

    ocr = (ocr_text or "").strip()
    qwen = _usable_page_text(qwen_text)
    if not qwen or _looks_like_json_dump(qwen):
        return ocr, False
    if not ocr:
        return qwen, True

    ocr_words = _content_words(ocr)
    qwen_words = _content_words(qwen)
    if not ocr_words:
        return qwen, True

    overlap = _folded_overlap(ocr, qwen)
    ocr_tr = _turkish_count(ocr)
    qwen_tr = _turkish_count(qwen)
    qwen_truncated = len(qwen) < len(ocr) * 0.72 and len(qwen_words) < len(ocr_words) * 0.72

    restored = _restore_turkish_from_vision(ocr, qwen)
    stitched = _stitch_missing_coverage(ocr, qwen)

    # Reading-order recovery: Qwen structure, only when it still covers OCR.
    if (
        corrupted
        and not qwen_truncated
        and overlap >= 0.32
        and (qwen_tr > ocr_tr or len(qwen.splitlines()) >= len(ocr.splitlines()) * 0.8)
        and _clause_starts(qwen) >= _clause_starts(ocr)
    ):
        return qwen, True

    if incomplete:
        if len(stitched) > len(ocr) * 1.02 or _turkish_count(stitched) > ocr_tr:
            chosen = _restore_turkish_from_vision(stitched, qwen)
            return chosen, True
        if not qwen_truncated and overlap >= 0.45 and len(qwen) >= len(ocr):
            return qwen, True

    if qwen_tr > ocr_tr + 2 and overlap >= 0.28:
        if _turkish_count(restored) > ocr_tr:
            return restored, True
        if not qwen_truncated:
            return qwen, True

    if corrupted and _turkish_count(restored) > ocr_tr:
        return restored, True

    return ocr, False


def _usable_page_text(text: str) -> str:
    """Strip a leaked JSON envelope so page.text is never a model dump."""

    candidate = (text or "").strip()
    if not candidate:
        return ""
    for _ in range(3):
        if not (candidate.startswith("{") and '"text"' in candidate):
            break
        parsed = _parse_vision_json(candidate)
        inner = str(parsed.get("text") or "").strip()
        if not inner or inner == candidate:
            break
        candidate = inner
    if _looks_like_json_dump(candidate):
        return ""
    return candidate


def _looks_like_json_dump(text: str) -> bool:
    stripped = (text or "").lstrip()
    return stripped.startswith("{") and '"signature"' in stripped


def _turkish_count(text: str) -> int:
    return sum(1 for c in text or "" if c in _TR_LETTERS)


def _fold_tr(text: str) -> str:
    return (text or "").translate(_TR_FOLD).casefold()


def _folded_overlap(ocr: str, qwen: str) -> float:
    ocr_words = {_fold_tr(w) for w in _content_words(ocr)}
    qwen_words = {_fold_tr(w) for w in _content_words(qwen)}
    if not ocr_words:
        return 1.0 if qwen_words else 0.0
    return len(ocr_words & qwen_words) / len(ocr_words)


def _clause_starts(text: str) -> int:
    return len(re.findall(r"(?m)^\s*\(\d+\)", text or ""))


def _restore_turkish_from_vision(ocr: str, qwen: str) -> str:
    """Replace ASCII-folded OCR tokens with Qwen's diacritic form when unique."""

    qwen_words = re.findall(r"[A-Za-z0-9À-ÿÇĞİÖŞÜçğıöşü']+", qwen or "", flags=re.UNICODE)
    by_fold: dict[str, list[str]] = {}
    for word in qwen_words:
        by_fold.setdefault(_fold_tr(word), []).append(word)

    unique: dict[str, str] = {}
    for key, vals in by_fold.items():
        turkish = [v for v in vals if any(c in _TR_LETTERS for c in v)]
        unique[key] = turkish[0] if turkish else vals[0]

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if any(c in _TR_LETTERS for c in token):
            return token
        mapped = unique.get(_fold_tr(token))
        if mapped and any(c in _TR_LETTERS for c in mapped):
            return mapped
        return token

    return re.sub(r"[A-Za-z0-9À-ÿÇĞİÖŞÜçğıöşü']+", replace_token, ocr)


def _stitch_missing_coverage(ocr: str, qwen: str) -> str:
    """Prepend / append Qwen lines that OCR never captured."""

    ocr_lines = [ln.strip() for ln in (ocr or "").splitlines() if ln.strip()]
    qwen_lines = [ln.strip() for ln in (qwen or "").splitlines() if ln.strip()]
    if not qwen_lines:
        return ocr
    ocr_keys = {_fold_tr(ln) for ln in ocr_lines}

    def covered(line: str) -> bool:
        key = _fold_tr(line)
        if key in ocr_keys:
            return True
        return any(
            key in ok or ok in key
            for ok in ocr_keys
            if min(len(key), len(ok)) >= 24
        )

    prefix: list[str] = []
    for line in qwen_lines:
        if covered(line):
            break
        prefix.append(line)

    suffix: list[str] = []
    for line in reversed(qwen_lines):
        if covered(line):
            break
        suffix.append(line)
    suffix.reverse()

    parts = []
    if prefix:
        parts.append("\n".join(prefix))
    parts.append(ocr.strip())
    if suffix:
        parts.append("\n".join(suffix))
    return "\n".join(p for p in parts if p).strip()


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9À-ÿÇĞİÖŞÜçğıöşü]{4,}", text or "", flags=re.UNICODE)
    return {w.casefold() for w in words}


def _encode_image_b64(image: np.ndarray, *, mime: str) -> str:
    import cv2

    if image is None or getattr(image, "size", 0) == 0:
        raise VisionFallbackError("Empty image received for vision fallback.")

    work = image
    h, w = work.shape[:2]
    max_edge = max(h, w)
    if max_edge > _QWEN_MAX_EDGE:
        scale = _QWEN_MAX_EDGE / max_edge
        work = cv2.resize(work, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if mime == "jpeg":
        ok, buf = cv2.imencode(".jpg", work, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    else:
        ok, buf = cv2.imencode(".png", work)
    if not ok:
        raise VisionFallbackError("Failed to encode page image for vision fallback.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _chat_completions(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    image_b64: str,
    mime: str,
    timeout_s: int,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:240]
        raise VisionFallbackError(f"Qwen-VL HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        if "timed out" in reason.lower():
            raise VisionFallbackError(f"Qwen-VL timed out after {timeout_s}s.") from exc
        raise VisionFallbackError(f"Qwen-VL unreachable at {endpoint}: {reason}") from exc
    except TimeoutError as exc:
        raise VisionFallbackError(f"Qwen-VL timed out after {timeout_s}s.") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisionFallbackError("Qwen-VL returned non-JSON response.") from exc

    choices = data.get("choices") or []
    if not choices:
        raise VisionFallbackError("Qwen-VL returned empty choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts).strip()
    if content is None:
        raise VisionFallbackError("Qwen-VL message content missing.")
    return str(content).strip()


def _parse_vision_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {"text": ""}
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            text = str(data.get("text") or "")
            data["text"] = _unescape_json_string(text)
            return data
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                text = str(data.get("text") or "")
                data["text"] = _unescape_json_string(text)
                return data
        except json.JSONDecodeError:
            pass
    extracted = _extract_json_text_field(cleaned)
    if extracted:
        signature = {"detected": False, "handwritten": False}
        stamp = {"detected": False}
        sig_m = re.search(r'"signature"\s*:\s*\{([^}]*)\}', cleaned)
        if sig_m:
            body = sig_m.group(1)
            signature["detected"] = bool(re.search(r'"detected"\s*:\s*true', body, flags=re.I))
            signature["handwritten"] = bool(re.search(r'"handwritten"\s*:\s*true', body, flags=re.I))
        stamp_m = re.search(r'"stamp"\s*:\s*\{([^}]*)\}', cleaned)
        if stamp_m:
            stamp["detected"] = bool(re.search(r'"detected"\s*:\s*true', stamp_m.group(1), flags=re.I))
        return {"text": extracted, "signature": signature, "stamp": stamp}
    if cleaned.lstrip().startswith("{") and "\"signature\"" in cleaned:
        return {
            "text": "",
            "signature": {"detected": False, "handwritten": False},
            "stamp": {"detected": False},
        }
    return {
        "text": cleaned,
        "signature": {"detected": False, "handwritten": False},
        "stamp": {"detected": False},
    }


def _extract_json_text_field(raw: str) -> str:
    match = re.search(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', raw, flags=re.S)
    if not match:
        return ""
    return _unescape_json_string(match.group(1)).strip()


def _unescape_json_string(text: str) -> str:
    return (
        text.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )
