"""Vision fallback (PaddleOCR-VL) — last resort after OCR retries fail a page."""

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

# PaddleOCR-VL GGUF / llama-server practical image edge (pixels).
_VISION_MAX_EDGE = 1792

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


class PaddleOCRVLVisionFallback(VisionFallbackInterface):
    """PaddleOCR-VL-1.6 (GGUF) via local OpenAI-compatible llama-server.

    Default endpoint: http://127.0.0.1:8111/v1/chat/completions
    (see Inference/start_paddleocr_vl.bat).
    """

    provider_name = "paddleocr-vl"

    def __init__(self, config: VisionFallbackConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client  # lazy: only resolved on first real call

    def _ensure_http_ready(self) -> str:
        endpoint = (self.config.endpoint or "").strip()
        if not endpoint:
            raise VisionFallbackError("OCR_VISION_FALLBACK_ENDPOINT is not configured.")
        if not endpoint.startswith("http"):
            raise VisionFallbackError(f"Invalid PaddleOCR-VL endpoint: {endpoint}")
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
                "using local PaddleOCR-VL HTTP endpoint.",
                exc,
            )
            return None

    def read_page(self, image: np.ndarray) -> VisionFallbackResult:
        """Transcribe an entire page with no application-level text-size cap.

        Tall pages are split into as many horizontal bands as needed (each band
        gets a full ``max_tokens`` budget). If a reply hits the API token
        ceiling, generation is continued until the band is finished.
        """
        if image is None or getattr(image, "shape", None) is None:
            raise VisionFallbackError("Empty image received for vision fallback.")
        height = int(image.shape[0])
        target = max(400, int(self.config.band_target_px))
        bands = max(1, (height + target - 1) // target)
        if bands == 1:
            return self._call_complete(image, _TEXT_AND_VISION_PROMPT)
        return self._read_page_banded(image, bands=bands)

    def inspect_visuals(self, image: np.ndarray) -> VisionFallbackResult:
        return self._call_complete(image, _VISION_ONLY_PROMPT, max_tokens=min(512, self.config.max_tokens))

    def _read_page_banded(self, image: np.ndarray, *, bands: int) -> VisionFallbackResult:
        height = int(image.shape[0])
        overlap = min(0.35, max(0.0, float(self.config.band_overlap)))
        band_h = height / bands
        overlap_px = int(band_h * overlap)
        parts: list[str] = []
        signature_detected = False
        signature_handwritten = False
        stamp_detected = False
        logger.info("[VISION] Banded page read bands=%s height=%s (full token budget per band)", bands, height)
        for index in range(bands):
            y0 = int(index * band_h) - (overlap_px if index > 0 else 0)
            y1 = int((index + 1) * band_h) + (overlap_px if index < bands - 1 else 0)
            y0 = max(0, y0)
            y1 = min(height, max(y0 + 1, y1))
            crop = image[y0:y1, :]
            # Full max_tokens per band — never divide the budget across bands.
            result = self._call_complete(crop, _TEXT_AND_VISION_PROMPT)
            if result.text.strip():
                parts.append(result.text.strip())
            signature_detected = signature_detected or result.signature_detected
            signature_handwritten = signature_handwritten or result.signature_handwritten
            stamp_detected = stamp_detected or result.stamp_detected
        merged = _merge_band_texts(parts)
        return VisionFallbackResult(
            text=merged,
            confidence=None,
            uncertain=True,
            provider=self.provider_name,
            signature_detected=signature_detected,
            signature_handwritten=signature_handwritten,
            stamp_detected=stamp_detected,
        )

    def _call_complete(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> VisionFallbackResult:
        """Call the VL server and auto-continue if the reply was cut by max_tokens."""
        budget = max_tokens if max_tokens is not None else self.config.max_tokens
        texts: list[str] = []
        signature: dict[str, Any] = {}
        stamp: dict[str, Any] = {}
        confidence = None
        active_prompt = prompt
        for round_idx in range(max(1, self.config.max_continue_rounds)):
            raw_text, finish_reason, confidence = self._raw_call(
                image, active_prompt, max_tokens=budget,
            )
            parsed = _parse_vision_json(raw_text)
            chunk = _usable_page_text(str(parsed.get("text") or ""))
            if not chunk and not texts:
                # First round may return visuals-only JSON with empty text.
                chunk = ""
            if chunk:
                texts.append(chunk)
            if isinstance(parsed.get("signature"), dict):
                signature = parsed["signature"]
            if isinstance(parsed.get("stamp"), dict):
                stamp = parsed["stamp"]
            if finish_reason != "length":
                break
            logger.info(
                "[VISION] Reply truncated (finish_reason=length); continuing round=%s",
                round_idx + 2,
            )
            tail = (chunk or (texts[-1] if texts else ""))[-400:]
            active_prompt = (
                "Continue the page transcription exactly where you stopped. "
                "Do not repeat previous lines. Preserve Turkish letters. "
                f"Last transcribed fragment ends with:\n<<<\n{tail}\n>>>\n"
                "Return JSON only:\n"
                '{"text":"<remaining text only>","signature":{"detected":false,"handwritten":false},"stamp":{"detected":false}}'
            )
        merged_text = _merge_band_texts(texts) if len(texts) > 1 else (texts[0] if texts else "")
        return VisionFallbackResult(
            text=merged_text,
            confidence=confidence,
            uncertain=True,
            provider=self.provider_name,
            signature_detected=bool(signature.get("detected")),
            signature_handwritten=bool(signature.get("handwritten")),
            stamp_detected=bool(stamp.get("detected")),
        )

    def _raw_call(
        self, image: np.ndarray, prompt: str, *, max_tokens: int,
    ) -> tuple[str, str, float | None]:
        shared = self._maybe_shared_client()
        logger.info("[VISION] Sending page to PaddleOCR-VL")
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
                finish_reason = str(getattr(response, "finish_reason", "stop") or "stop")
            else:
                endpoint = self._ensure_http_ready()
                raw_text, finish_reason = _chat_completions(
                    endpoint=endpoint,
                    model=self.config.model_name,
                    prompt=prompt,
                    image_b64=_encode_image_b64(image, mime="png"),
                    mime="image/png",
                    timeout_s=self.config.request_timeout_s,
                    max_tokens=max_tokens,
                )
                confidence = None
        except VisionFallbackError:
            raise
        except Exception as exc:
            raise VisionFallbackError(f"PaddleOCR-VL fallback call failed: {exc}") from exc

        logger.info("[VISION] Response received finish_reason=%s", finish_reason)
        return raw_text, finish_reason, confidence

def build_vision_fallback(config: VisionFallbackConfig) -> VisionFallbackInterface:
    if not config.enabled:
        return NullVisionFallback()
    if config.provider in {"paddleocr-vl", "paddleocr_vl", "paddle"}:
        return PaddleOCRVLVisionFallback(config)
    logger.warning("Unknown vision fallback provider '%s'; disabling fallback.", config.provider)
    return NullVisionFallback()


def choose_better_text(ocr_text: str, vision_text: str) -> tuple[str, bool]:
    """Keep OCR unless vision recovers missing/corrupted regions."""

    return merge_ocr_and_vision(ocr_text, vision_text)


def merge_ocr_and_vision(
    ocr_text: str,
    vision_text: str,
    *,
    incomplete: bool = False,
    corrupted: bool = False,
) -> tuple[str, bool]:
    """Use vision fallback to fill gaps / verify spelling. Do not blindly replace good OCR."""

    ocr = (ocr_text or "").strip()
    vision = _usable_page_text(vision_text)
    if not vision or _looks_like_json_dump(vision):
        return ocr, False
    if not ocr:
        return vision, True

    ocr_words = _content_words(ocr)
    vision_words = _content_words(vision)
    if not ocr_words:
        return vision, True

    # When OCR is truncated/corrupted and vision recovered a clearly fuller page,
    # prefer vision (or a stitch) instead of keeping a short OCR fragment.
    if (incomplete or corrupted) and len(vision_words) >= max(40, int(len(ocr_words) * 1.25)):
        stitched = _stitch_missing_coverage(ocr, vision)
        if len(_content_words(stitched)) >= len(vision_words):
            return stitched, True
        return vision, True

    overlap = _folded_overlap(ocr, vision)
    ocr_tr = _turkish_count(ocr)
    vision_tr = _turkish_count(vision)
    vision_truncated = len(vision) < len(ocr) * 0.72 and len(vision_words) < len(ocr_words) * 0.72

    restored = _restore_turkish_from_vision(ocr, vision)
    stitched = _stitch_missing_coverage(ocr, vision)

    # Reading-order recovery: vision structure, only when it still covers OCR.
    if (
        corrupted
        and not vision_truncated
        and overlap >= 0.32
        and (vision_tr > ocr_tr or len(vision.splitlines()) >= len(ocr.splitlines()) * 0.8)
        and _clause_starts(vision) >= _clause_starts(ocr)
    ):
        return vision, True

    if incomplete:
        if len(stitched) > len(ocr) * 1.02 or _turkish_count(stitched) > ocr_tr:
            chosen = _restore_turkish_from_vision(stitched, vision)
            return chosen, True
        if not vision_truncated and overlap >= 0.45 and len(vision) >= len(ocr):
            return vision, True

    if vision_tr > ocr_tr + 2 and overlap >= 0.28:
        if _turkish_count(restored) > ocr_tr:
            return restored, True
        if not vision_truncated:
            return vision, True

    if corrupted and _turkish_count(restored) > ocr_tr:
        return restored, True

    return ocr, False


def _merge_band_texts(parts: list[str]) -> str:
    """Concatenate band transcriptions while dropping overlapping duplicate lines."""
    cleaned = [_usable_page_text(p) for p in parts if (p or "").strip()]
    cleaned = [p for p in cleaned if p]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]

    merged_lines: list[str] = []
    seen_folds: set[str] = set()
    for part in cleaned:
        lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
        for line in lines:
            key = _fold_tr(line)
            if len(key) >= 24 and any(key in s or s in key for s in seen_folds if len(s) >= 24):
                continue
            if key in seen_folds:
                continue
            seen_folds.add(key)
            merged_lines.append(line)
    return "\n".join(merged_lines).strip()


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


def _folded_overlap(ocr: str, vision: str) -> float:
    ocr_words = {_fold_tr(w) for w in _content_words(ocr)}
    vision_words = {_fold_tr(w) for w in _content_words(vision)}
    if not ocr_words:
        return 1.0 if vision_words else 0.0
    return len(ocr_words & vision_words) / len(ocr_words)


def _clause_starts(text: str) -> int:
    return len(re.findall(r"(?m)^\s*\(\d+\)", text or ""))


def _restore_turkish_from_vision(ocr: str, vision: str) -> str:
    """Replace ASCII-folded OCR tokens with vision's diacritic form when unique."""

    vision_words = re.findall(r"[A-Za-z0-9À-ÿÇĞİÖŞÜçğıöşü']+", vision or "", flags=re.UNICODE)
    by_fold: dict[str, list[str]] = {}
    for word in vision_words:
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


def _stitch_missing_coverage(ocr: str, vision: str) -> str:
    """Prepend / append vision lines that OCR never captured."""

    ocr_lines = [ln.strip() for ln in (ocr or "").splitlines() if ln.strip()]
    vision_lines = [ln.strip() for ln in (vision or "").splitlines() if ln.strip()]
    if not vision_lines:
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
    for line in vision_lines:
        if covered(line):
            break
        prefix.append(line)

    suffix: list[str] = []
    for line in reversed(vision_lines):
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
    if max_edge > _VISION_MAX_EDGE:
        scale = _VISION_MAX_EDGE / max_edge
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
) -> tuple[str, str]:
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
        raise VisionFallbackError(f"PaddleOCR-VL HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        if "timed out" in reason.lower():
            raise VisionFallbackError(f"PaddleOCR-VL timed out after {timeout_s}s.") from exc
        raise VisionFallbackError(f"PaddleOCR-VL unreachable at {endpoint}: {reason}") from exc
    except TimeoutError as exc:
        raise VisionFallbackError(f"PaddleOCR-VL timed out after {timeout_s}s.") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisionFallbackError("PaddleOCR-VL returned non-JSON response.") from exc

    choices = data.get("choices") or []
    if not choices:
        raise VisionFallbackError("PaddleOCR-VL returned empty choices.")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    finish_reason = str(choice.get("finish_reason") or "stop")
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts).strip(), finish_reason
    if content is None:
        raise VisionFallbackError("PaddleOCR-VL message content missing.")
    return str(content).strip(), finish_reason


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
