"""PaddleOCR-VL vision fallback for the OCR Agent."""

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
    "Transcribe ALL visible printed text on this document image.\n"
    "Start from the first visible line and continue to the last visible line.\n"
    "Include headers, body text, tables, margins, footers, and other printed text.\n"
    "\n"
    "Rules:\n"
    "- Copy the text exactly.\n"
    "- Do not summarize.\n"
    "- Do not translate.\n"
    "- Do not rewrite.\n"
    "- Do not invent unreadable text.\n"
    "- Preserve Turkish characters exactly: İ ı Ş ş Ğ ğ Ç ç Ö ö Ü ü.\n"
    "- Preserve paragraphs and line breaks when possible.\n"
    "- Reading order: top-to-bottom, left-to-right.\n"
    "- For multi-column text, finish the left column before the right column.\n"
    "- Printed words such as İmza, signature, mühür, or stamp are not evidence\n"
    "  of a handwritten signature or physical stamp.\n"
    "- CamScanner watermarks are not stamps.\n"
    "- Logos are not signatures.\n"
    "\n"
    "Return only the transcription text."
)


_VISION_ONLY_PROMPT = (
    "Inspect this document image only for visible physical marks.\n"
    "\n"
    "Determine:\n"
    "- signature_detected: true only when a handwritten or ink signature is visible.\n"
    "- signature_handwritten: true only when the detected signature is handwritten.\n"
    "- stamp_detected: true only when a physical stamp, seal, or mühür is visible.\n"
    "\n"
    "Printed words such as İmza, signature, mühür, or stamp are not evidence.\n"
    "CamScanner watermarks are not stamps.\n"
    "Logos are not signatures.\n"
    "\n"
    "Return JSON only:\n"
    '{"signature_detected":false,"signature_handwritten":false,"stamp_detected":false}'
)


_VISION_MAX_EDGE = 1792


_TR_FOLD = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "I": "i",
        "í": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "c",
        "Ğ": "g",
        "İ": "i",
        "Ö": "o",
        "Ş": "s",
        "Ü": "u",
    }
)


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
    provider_name: str = "unknown"

    @abstractmethod
    def read_page(self, image: np.ndarray) -> VisionFallbackResult:
        raise NotImplementedError

    def inspect_visuals(self, image: np.ndarray) -> VisionFallbackResult:
        return self.read_page(image)


class NullVisionFallback(VisionFallbackInterface):
    provider_name = "disabled"

    def read_page(self, image: np.ndarray) -> VisionFallbackResult:
        raise VisionFallbackError(
            "Vision fallback was requested but is disabled or unconfigured."
        )

    def inspect_visuals(self, image: np.ndarray) -> VisionFallbackResult:
        raise VisionFallbackError("Vision fallback is disabled.")


class PaddleOCRVLVisionFallback(VisionFallbackInterface):
    provider_name = "paddleocr-vl"

    def __init__(
        self,
        config: VisionFallbackConfig,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self._client = client

    def _ensure_http_ready(self) -> str:
        endpoint = (self.config.endpoint or "").strip()

        if not endpoint:
            raise VisionFallbackError(
                "OCR_VISION_FALLBACK_ENDPOINT is not configured."
            )

        if not endpoint.startswith(("http://", "https://")):
            raise VisionFallbackError(
                f"Invalid PaddleOCR-VL endpoint: {endpoint}"
            )

        return endpoint

    def _maybe_shared_client(self) -> Any | None:
        if self._client is not None:
            return self._client

        if not self.config.use_shared_inference_client:
            return None

        try:
            from Inference.client.llama_client import VisionInferenceClient

            self._client = VisionInferenceClient()
            return self._client

        except Exception as exc:
            logger.warning(
                "Shared vision inference client unavailable: %s. "
                "Using local PaddleOCR-VL HTTP endpoint.",
                exc,
            )
            return None

    def read_page(self, image: np.ndarray) -> VisionFallbackResult:
        if image is None:
            raise VisionFallbackError(
                "Empty image received for vision fallback."
            )

        if getattr(image, "size", 0) == 0:
            raise VisionFallbackError(
                "Empty image received for vision fallback."
            )

        return self._call_complete(
            image=image,
            prompt=_TEXT_AND_VISION_PROMPT,
        )

    def inspect_visuals(
        self,
        image: np.ndarray,
    ) -> VisionFallbackResult:
        if image is None:
            raise VisionFallbackError(
                "Empty image received for visual inspection."
            )

        if getattr(image, "size", 0) == 0:
            raise VisionFallbackError(
                "Empty image received for visual inspection."
            )

        return self._call_complete(
            image=image,
            prompt=_VISION_ONLY_PROMPT,
            max_tokens=min(512, self.config.max_tokens),
            parse_visuals_only=True,
        )

    def _call_complete(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        max_tokens: int | None = None,
        parse_visuals_only: bool = False,
    ) -> VisionFallbackResult:
        budget = (
            max_tokens
            if max_tokens is not None
            else self.config.max_tokens
        )

        texts: list[str] = []

        signature_detected = False
        signature_handwritten = False
        stamp_detected = False

        confidence: float | None = None

        active_prompt = prompt

        rounds = max(
            1,
            int(self.config.max_continue_rounds),
        )

        for round_index in range(rounds):
            raw_text, finish_reason, confidence = self._raw_call(
                image=image,
                prompt=active_prompt,
                max_tokens=budget,
            )

            if parse_visuals_only:
                parsed = _parse_visuals_json(raw_text)

                signature_detected = bool(
                    parsed.get("signature_detected")
                )

                signature_handwritten = bool(
                    parsed.get("signature_handwritten")
                )

                stamp_detected = bool(
                    parsed.get("stamp_detected")
                )

                break

            text = _extract_transcription(raw_text)

            if text:
                texts.append(text)

            parsed = _parse_optional_visuals(raw_text)

            signature_detected = (
                signature_detected
                or bool(parsed.get("signature_detected"))
            )

            signature_handwritten = (
                signature_handwritten
                or bool(parsed.get("signature_handwritten"))
            )

            stamp_detected = (
                stamp_detected
                or bool(parsed.get("stamp_detected"))
            )

            if finish_reason != "length":
                break

            if round_index + 1 >= rounds:
                break

            tail = (
                text
                or (texts[-1] if texts else "")
            )[-500:]

            active_prompt = (
                "Continue the transcription from exactly where the "
                "previous response stopped.\n"
                "Do not repeat text already transcribed.\n"
                "Do not summarize or translate.\n"
                "Preserve Turkish characters exactly.\n"
                "Return only the remaining transcription text.\n"
                "\n"
                "Last transcribed fragment:\n"
                "<<<\n"
                f"{tail}\n"
                ">>>\n"
            )

            logger.info(
                "[VISION] Response truncated; continuing round %s/%s",
                round_index + 2,
                rounds,
            )

        merged_text = _merge_text_chunks(texts)

        return VisionFallbackResult(
            text=merged_text,
            confidence=confidence,
            uncertain=True,
            provider=self.provider_name,
            signature_detected=signature_detected,
            signature_handwritten=signature_handwritten,
            stamp_detected=stamp_detected,
        )

    def _raw_call(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        max_tokens: int,
    ) -> tuple[str, str, float | None]:
        shared = self._maybe_shared_client()

        logger.info(
            "[VISION] Sending page to PaddleOCR-VL"
        )

        try:
            if (
                shared is not None
                and hasattr(shared, "generate_vision")
            ):
                image_b64 = _encode_image_b64(
                    image,
                    mime="png",
                )

                response = shared.generate_vision(
                    image_base64=image_b64,
                    instructions=prompt,
                    timeout_s=self.config.request_timeout_s,
                )

                raw_text = str(
                    getattr(response, "text", response) or ""
                ).strip()

                confidence = getattr(
                    response,
                    "confidence",
                    None,
                )

                finish_reason = str(
                    getattr(
                        response,
                        "finish_reason",
                        "stop",
                    )
                    or "stop"
                )

                return (
                    raw_text,
                    finish_reason,
                    confidence,
                )

            endpoint = self._ensure_http_ready()

            raw_text, finish_reason = _chat_completions(
                endpoint=endpoint,
                model=self.config.model_name,
                prompt=prompt,
                image_b64=_encode_image_b64(
                    image,
                    mime="png",
                ),
                mime="image/png",
                timeout_s=self.config.request_timeout_s,
                max_tokens=max_tokens,
            )

            return raw_text, finish_reason, None

        except VisionFallbackError:
            raise

        except Exception as exc:
            raise VisionFallbackError(
                f"PaddleOCR-VL fallback call failed: {exc}"
            ) from exc


def build_vision_fallback(
    config: VisionFallbackConfig,
) -> VisionFallbackInterface:
    if not config.enabled:
        return NullVisionFallback()

    provider = (
        config.provider or ""
    ).strip().lower()

    if provider in {
        "paddleocr-vl",
        "paddleocr_vl",
        "paddle",
    }:
        return PaddleOCRVLVisionFallback(config)

    logger.warning(
        "Unknown vision fallback provider '%s'. "
        "Vision fallback disabled.",
        config.provider,
    )

    return NullVisionFallback()


def choose_better_text(
    ocr_text: str,
    vision_text: str,
) -> tuple[str, bool]:
    return merge_ocr_and_vision(
        ocr_text,
        vision_text,
    )


def merge_ocr_and_vision(
    ocr_text: str,
    vision_text: str,
    *,
    incomplete: bool = False,
    corrupted: bool = False,
) -> tuple[str, bool]:
    ocr = (ocr_text or "").strip()
    vision = _usable_page_text(vision_text)

    if not vision:
        return ocr, False

    if not ocr:
        return vision, True

    ocr_words = _content_words(ocr)
    vision_words = _content_words(vision)

    if not ocr_words:
        return vision, True

    if (
        (incomplete or corrupted)
        and len(vision_words)
        >= max(
            40,
            int(len(ocr_words) * 1.25),
        )
    ):
        stitched = _stitch_missing_coverage(
            ocr,
            vision,
        )

        if len(_content_words(stitched)) >= len(vision_words):
            return stitched, True

        return vision, True

    overlap = _folded_overlap(
        ocr,
        vision,
    )

    ocr_tr = _turkish_count(ocr)
    vision_tr = _turkish_count(vision)

    vision_truncated = (
        len(vision) < len(ocr) * 0.72
        and len(vision_words)
        < len(ocr_words) * 0.72
    )

    restored = _restore_turkish_from_vision(
        ocr,
        vision,
    )

    stitched = _stitch_missing_coverage(
        ocr,
        vision,
    )

    if (
        corrupted
        and not vision_truncated
        and overlap >= 0.32
        and (
            vision_tr > ocr_tr
            or len(vision.splitlines())
            >= len(ocr.splitlines()) * 0.8
        )
        and _clause_starts(vision)
        >= _clause_starts(ocr)
    ):
        return vision, True

    if incomplete:
        if (
            len(stitched) > len(ocr) * 1.02
            or _turkish_count(stitched) > ocr_tr
        ):
            chosen = _restore_turkish_from_vision(
                stitched,
                vision,
            )

            return chosen, True

        if (
            not vision_truncated
            and overlap >= 0.45
            and len(vision) >= len(ocr)
        ):
            return vision, True

    if (
        vision_tr > ocr_tr + 2
        and overlap >= 0.28
    ):
        if _turkish_count(restored) > ocr_tr:
            return restored, True

        if not vision_truncated:
            return vision, True

    if (
        corrupted
        and _turkish_count(restored) > ocr_tr
    ):
        return restored, True

    return ocr, False


def _merge_text_chunks(
    chunks: list[str],
) -> str:
    cleaned = [
        _usable_page_text(chunk)
        for chunk in chunks
        if chunk
    ]

    cleaned = [
        chunk
        for chunk in cleaned
        if chunk
    ]

    if not cleaned:
        return ""

    if len(cleaned) == 1:
        return cleaned[0]

    merged: list[str] = []
    seen: set[str] = set()

    for chunk in cleaned:
        lines = [
            line.strip()
            for line in chunk.splitlines()
            if line.strip()
        ]

        for line in lines:
            key = _fold_tr(line)

            if not key:
                continue

            if key in seen:
                continue

            duplicate = False

            if len(key) >= 24:
                for existing in seen:
                    if len(existing) < 24:
                        continue

                    if (
                        key in existing
                        or existing in key
                    ):
                        duplicate = True
                        break

            if duplicate:
                continue

            seen.add(key)
            merged.append(line)

    return "\n".join(merged).strip()


def _extract_transcription(
    raw_text: str,
) -> str:
    if not raw_text:
        return ""

    cleaned = raw_text.strip()

    parsed = _parse_vision_json(cleaned)

    if "text" in parsed:
        return _usable_page_text(
            str(parsed.get("text") or "")
        )

    return _usable_page_text(cleaned)


def _usable_page_text(
    text: str,
) -> str:
    candidate = (text or "").strip()

    if not candidate:
        return ""

    for _ in range(3):
        if not (
            candidate.startswith("{")
            and '"text"' in candidate
        ):
            break

        parsed = _parse_vision_json(
            candidate
        )

        inner = str(
            parsed.get("text") or ""
        ).strip()

        if not inner or inner == candidate:
            break

        candidate = inner

    if _looks_like_json_dump(candidate):
        return ""

    return candidate


def _looks_like_json_dump(
    text: str,
) -> bool:
    stripped = (text or "").lstrip()

    return (
        stripped.startswith("{")
        and (
            '"signature"' in stripped
            or '"stamp"' in stripped
            or '"signature_detected"' in stripped
        )
    )


def _parse_vision_json(
    raw: str,
) -> dict[str, Any]:
    if not raw:
        return {"text": ""}

    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    try:
        data = json.loads(cleaned)

        if isinstance(data, dict):
            if "text" in data:
                data["text"] = _unescape_json_string(
                    str(data.get("text") or "")
                )

            return data

    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_RE.search(cleaned)

    if match:
        try:
            data = json.loads(
                match.group(0)
            )

            if isinstance(data, dict):
                if "text" in data:
                    data["text"] = _unescape_json_string(
                        str(data.get("text") or "")
                    )

                return data

        except json.JSONDecodeError:
            pass

    extracted = _extract_json_text_field(
        cleaned
    )

    if extracted:
        return {
            "text": extracted,
            **_parse_optional_visuals(cleaned),
        }

    return {
        "text": cleaned,
        **_parse_optional_visuals(cleaned),
    }


def _parse_optional_visuals(
    raw: str,
) -> dict[str, bool]:
    signature_detected = False
    signature_handwritten = False
    stamp_detected = False

    if not raw:
        return {
            "signature_detected": False,
            "signature_handwritten": False,
            "stamp_detected": False,
        }

    try:
        data = json.loads(raw)

        if isinstance(data, dict):
            signature = data.get("signature")

            if isinstance(signature, dict):
                signature_detected = bool(
                    signature.get("detected")
                )

                signature_handwritten = bool(
                    signature.get("handwritten")
                )

            stamp = data.get("stamp")

            if isinstance(stamp, dict):
                stamp_detected = bool(
                    stamp.get("detected")
                )

            signature_detected = (
                signature_detected
                or bool(
                    data.get("signature_detected")
                )
            )

            signature_handwritten = (
                signature_handwritten
                or bool(
                    data.get("signature_handwritten")
                )
            )

            stamp_detected = (
                stamp_detected
                or bool(
                    data.get("stamp_detected")
                )
            )

            return {
                "signature_detected": signature_detected,
                "signature_handwritten": signature_handwritten,
                "stamp_detected": stamp_detected,
            }

    except (json.JSONDecodeError, TypeError):
        pass

    signature_match = re.search(
        r'"signature"\s*:\s*\{([^}]*)\}',
        raw,
        flags=re.S,
    )

    if signature_match:
        body = signature_match.group(1)

        signature_detected = bool(
            re.search(
                r'"detected"\s*:\s*true',
                body,
                flags=re.I,
            )
        )

        signature_handwritten = bool(
            re.search(
                r'"handwritten"\s*:\s*true',
                body,
                flags=re.I,
            )
        )

    stamp_match = re.search(
        r'"stamp"\s*:\s*\{([^}]*)\}',
        raw,
        flags=re.S,
    )

    if stamp_match:
        stamp_detected = bool(
            re.search(
                r'"detected"\s*:\s*true',
                stamp_match.group(1),
                flags=re.I,
            )
        )

    signature_detected = (
        signature_detected
        or bool(
            re.search(
                r'"signature_detected"\s*:\s*true',
                raw,
                flags=re.I,
            )
        )
    )

    signature_handwritten = (
        signature_handwritten
        or bool(
            re.search(
                r'"signature_handwritten"\s*:\s*true',
                raw,
                flags=re.I,
            )
        )
    )

    stamp_detected = (
        stamp_detected
        or bool(
            re.search(
                r'"stamp_detected"\s*:\s*true',
                raw,
                flags=re.I,
            )
        )
    )

    return {
        "signature_detected": signature_detected,
        "signature_handwritten": signature_handwritten,
        "stamp_detected": stamp_detected,
    }


def _parse_visuals_json(
    raw: str,
) -> dict[str, bool]:
    parsed = _parse_optional_visuals(raw)

    return {
        "signature_detected": bool(
            parsed.get("signature_detected")
        ),
        "signature_handwritten": bool(
            parsed.get("signature_handwritten")
        ),
        "stamp_detected": bool(
            parsed.get("stamp_detected")
        ),
    }


def _extract_json_text_field(
    raw: str,
) -> str:
    match = re.search(
        r'"text"\s*:\s*"((?:\\.|[^"\\])*)"',
        raw,
        flags=re.S,
    )

    if not match:
        return ""

    return _unescape_json_string(
        match.group(1)
    ).strip()


def _unescape_json_string(
    text: str,
) -> str:
    return (
        text.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def _turkish_count(
    text: str,
) -> int:
    return sum(
        1
        for char in text or ""
        if char in _TR_LETTERS
    )


def _fold_tr(
    text: str,
) -> str:
    return (
        text or ""
    ).translate(_TR_FOLD).casefold()


def _folded_overlap(
    ocr: str,
    vision: str,
) -> float:
    ocr_words = {
        _fold_tr(word)
        for word in _content_words(ocr)
    }

    vision_words = {
        _fold_tr(word)
        for word in _content_words(vision)
    }

    if not ocr_words:
        return (
            1.0
            if vision_words
            else 0.0
        )

    return (
        len(ocr_words & vision_words)
        / len(ocr_words)
    )


def _clause_starts(
    text: str,
) -> int:
    return len(
        re.findall(
            r"(?m)^\s*\(\d+\)",
            text or "",
        )
    )


def _restore_turkish_from_vision(
    ocr: str,
    vision: str,
) -> str:
    vision_words = re.findall(
        r"[A-Za-z0-9À-ÿÇĞİÖŞÜçğıöşü']+",
        vision or "",
        flags=re.UNICODE,
    )

    by_fold: dict[str, list[str]] = {}

    for word in vision_words:
        by_fold.setdefault(
            _fold_tr(word),
            [],
        ).append(word)

    unique: dict[str, str] = {}

    for key, values in by_fold.items():
        turkish = [
            value
            for value in values
            if any(
                char in _TR_LETTERS
                for char in value
            )
        ]

        unique[key] = (
            turkish[0]
            if turkish
            else values[0]
        )

    def replace_token(
        match: re.Match[str],
    ) -> str:
        token = match.group(0)

        if any(
            char in _TR_LETTERS
            for char in token
        ):
            return token

        mapped = unique.get(
            _fold_tr(token)
        )

        if (
            mapped
            and any(
                char in _TR_LETTERS
                for char in mapped
            )
        ):
            return mapped

        return token

    return re.sub(
        r"[A-Za-z0-9À-ÿÇĞİÖŞÜçğıöşü']+",
        replace_token,
        ocr,
    )


def _stitch_missing_coverage(
    ocr: str,
    vision: str,
) -> str:
    ocr_lines = [
        line.strip()
        for line in (ocr or "").splitlines()
        if line.strip()
    ]

    vision_lines = [
        line.strip()
        for line in (vision or "").splitlines()
        if line.strip()
    ]

    if not vision_lines:
        return ocr

    ocr_keys = {
        _fold_tr(line)
        for line in ocr_lines
    }

    def covered(
        line: str,
    ) -> bool:
        key = _fold_tr(line)

        if key in ocr_keys:
            return True

        return any(
            key in existing
            or existing in key
            for existing in ocr_keys
            if min(
                len(key),
                len(existing),
            ) >= 24
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

    parts: list[str] = []

    if prefix:
        parts.append(
            "\n".join(prefix)
        )

    if ocr.strip():
        parts.append(
            ocr.strip()
        )

    if suffix:
        parts.append(
            "\n".join(suffix)
        )

    return "\n".join(
        part
        for part in parts
        if part
    ).strip()


def _content_words(
    text: str,
) -> set[str]:
    words = re.findall(
        r"[A-Za-z0-9À-ÿÇĞİÖŞÜçğıöşü]{4,}",
        text or "",
        flags=re.UNICODE,
    )

    return {
        word.casefold()
        for word in words
    }


def _encode_image_b64(
    image: np.ndarray,
    *,
    mime: str,
) -> str:
    import cv2

    if image is None:
        raise VisionFallbackError(
            "Empty image received for vision fallback."
        )

    if getattr(image, "size", 0) == 0:
        raise VisionFallbackError(
            "Empty image received for vision fallback."
        )

    work = image

    height, width = work.shape[:2]
    max_edge = max(
        height,
        width,
    )

    if max_edge > _VISION_MAX_EDGE:
        scale = (
            _VISION_MAX_EDGE
            / max_edge
        )

        new_width = max(
            1,
            int(width * scale),
        )

        new_height = max(
            1,
            int(height * scale),
        )

        work = cv2.resize(
            work,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

    if mime == "jpeg":
        ok, buffer = cv2.imencode(
            ".jpg",
            work,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                88,
            ],
        )
    else:
        ok, buffer = cv2.imencode(
            ".png",
            work,
        )

    if not ok:
        raise VisionFallbackError(
            "Failed to encode page image."
        )

    return base64.b64encode(
        buffer.tobytes()
    ).decode("ascii")


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
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime};base64,"
                                f"{image_b64}"
                            )
                        },
                    },
                ],
            }
        ],
    }

    body = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(
            req,
            timeout=timeout_s,
        ) as response:
            raw = response.read().decode(
                "utf-8",
                errors="replace",
            )

    except error.HTTPError as exc:
        detail = exc.read().decode(
            "utf-8",
            errors="replace",
        )[:1000]

        raise VisionFallbackError(
            f"PaddleOCR-VL HTTP {exc.code}: {detail}"
        ) from exc

    except error.URLError as exc:
        reason = str(
            getattr(
                exc,
                "reason",
                exc,
            )
        )

        if "timed out" in reason.lower():
            raise VisionFallbackError(
                f"PaddleOCR-VL timed out after {timeout_s}s."
            ) from exc

        raise VisionFallbackError(
            f"PaddleOCR-VL unreachable at "
            f"{endpoint}: {reason}"
        ) from exc

    except TimeoutError as exc:
        raise VisionFallbackError(
            f"PaddleOCR-VL timed out after {timeout_s}s."
        ) from exc

    try:
        data = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise VisionFallbackError(
            "PaddleOCR-VL returned a non-JSON response."
        ) from exc

    choices = data.get("choices") or []

    if not choices:
        raise VisionFallbackError(
            "PaddleOCR-VL returned empty choices."
        )

    choice = (
        choices[0]
        if isinstance(
            choices[0],
            dict,
        )
        else {}
    )

    finish_reason = str(
        choice.get("finish_reason")
        or "stop"
    )

    message = (
        choice.get("message")
        or {}
    )

    content = message.get(
        "content"
    )

    if isinstance(content, list):
        parts: list[str] = []

        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
            ):
                parts.append(
                    str(
                        part.get("text")
                        or ""
                    )
                )

            elif isinstance(part, str):
                parts.append(part)

        return (
            "\n".join(parts).strip(),
            finish_reason,
        )

    if content is None:
        raise VisionFallbackError(
            "PaddleOCR-VL message content is missing."
        )

    return (
        str(content).strip(),
        finish_reason,
    )