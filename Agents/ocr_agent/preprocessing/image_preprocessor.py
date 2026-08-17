from __future__ import annotations

import cv2
import numpy as np

from ..config import PreprocessingConfig
from ..exceptions import PreprocessingError


class ImagePreprocessor:
    """Prepare hard phone photos: far, tilted, pale/faded documents."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def process(self, image: np.ndarray, strength: str = "full") -> np.ndarray:
        if image is None or image.size == 0:
            raise PreprocessingError("Empty image received.")
        try:
            out = image.copy()
            if out.ndim == 2:
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

            if strength == "light":
                out = self._ensure_min_dimension(out)
                if self.config.max_dimension > 0:
                    out = self._resize_max(out)
                if self.config.deskew:
                    out = self._deskew(out)
                return out

            # 1) Far photo: pull the page out of a busy background
            if self.config.auto_crop_document:
                out = self._auto_crop_document(out)

            # 2) Tilt / keystone
            if self.config.perspective_correction:
                out = self._perspective_correct(out)

            # 3) Far / small text: enlarge before OCR
            out = self._ensure_min_dimension(out)
            if self.config.max_dimension > 0:
                out = self._resize_max(out)

            # 4) Pale / faded colors
            if self.config.pale_boost or self.config.contrast:
                out = self._boost_pale_contrast(out)

            if self.config.grayscale:
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if self.config.denoise:
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                h = max(3, int(self.config.denoise_strength))
                gray = cv2.fastNlMeansDenoising(gray, None, h, 7, 21)
                out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if self.config.sharpen:
                blur = cv2.GaussianBlur(out, (0, 0), 1.0)
                out = cv2.addWeighted(
                    out,
                    1.0 + self.config.sharpen_amount,
                    blur,
                    -self.config.sharpen_amount,
                    0,
                )

            # 5) Residual rotation after perspective
            if self.config.deskew:
                out = self._deskew(out)

            if self.config.adaptive_threshold:
                out = self._adaptive_binarize(out)

            return out
        except Exception as exc:
            raise PreprocessingError(f"Image preprocessing failed: {exc}") from exc

    def _resize_max(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.config.max_dimension:
            return image
        scale = self.config.max_dimension / max_dim
        return cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    def _ensure_min_dimension(self, image: np.ndarray) -> np.ndarray:
        min_dim = int(self.config.min_dimension or 0)
        if min_dim <= 0:
            return image
        h, w = image.shape[:2]
        short = min(h, w)
        if short >= min_dim:
            return image
        scale = min_dim / short
        return cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    def _auto_crop_document(self, image: np.ndarray) -> np.ndarray:
        """Crop around the largest bright page region (phone photos from afar)."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Page is usually brighter than desk / background
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(np.mean(mask == 255)) < 0.15:
            mask = cv2.bitwise_not(mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(contour) / float(h * w)
        if area_ratio < 0.12 or area_ratio > 0.85:
            return image

        x, y, bw, bh = cv2.boundingRect(contour)
        pad = int(0.05 * max(h, w))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        cropped = image[y0:y1, x0:x1]
        if cropped.size == 0:
            return image
        return cropped

    def _boost_pale_contrast(self, image: np.ndarray) -> np.ndarray:
        """Recover washed-out / low-contrast scans and pale phone photos."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = 3.5 if self.config.pale_boost else 2.0
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        if self.config.pale_boost:
            # Percentile stretch on luminance to fight faded ink/paper
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            lo, hi = np.percentile(gray, (2, 98))
            if hi > lo + 1:
                scale = 255.0 / (hi - lo)
                stretched = np.clip((gray.astype(np.float32) - lo) * scale, 0, 255).astype(
                    np.uint8
                )
                # Soft gamma for pale midtones
                gamma = 0.85
                table = np.array(
                    [((i / 255.0) ** gamma) * 255 for i in range(256)],
                    dtype=np.uint8,
                )
                stretched = cv2.LUT(stretched, table)
                out = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
        return out

    def _adaptive_binarize(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            12,
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _perspective_correct(self, image: np.ndarray) -> np.ndarray:
        # Conservative correction: only transform when a strong four-corner contour exists.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 140)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image.shape[:2]
        area_threshold = 0.25 * h * w

        best = None
        best_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < area_threshold or area < best_area:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                best = approx.reshape(4, 2).astype(np.float32)
                best_area = area

        if best is None:
            # Fallback: Otsu page mask quad
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if float(np.mean(mask == 255)) < 0.2:
                mask = cv2.bitwise_not(mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4 and cv2.contourArea(approx) >= area_threshold:
                    best = approx.reshape(4, 2).astype(np.float32)

        if best is None:
            return image

        ordered = self._order_points(best)
        (tl, tr, br, bl) = ordered
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_w = max(int(width_a), int(width_b))
        max_h = max(int(height_a), int(height_b))
        if max_w < 120 or max_h < 120:
            return image

        dst = np.array(
            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(
            image, matrix, (max_w, max_h), borderMode=cv2.BORDER_REPLICATE
        )

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1).ravel()
        return np.array(
            [
                points[np.argmin(s)],
                points[np.argmin(diff)],
                points[np.argmax(s)],
                points[np.argmax(diff)],
            ],
            dtype=np.float32,
        )

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        angle = self._estimate_skew_angle(image)
        if abs(angle) < self.config.deskew_min_angle or abs(angle) > self.config.deskew_max_angle:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _estimate_skew_angle(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Prefer text-line Hough estimate for phone photos
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=80,
            minLineLength=max(40, image.shape[1] // 12),
            maxLineGap=20,
        )
        angles: list[float] = []
        if lines is not None:
            for raw in lines:
                pts = np.asarray(raw).reshape(-1)
                if pts.size < 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in pts[:4]]
                if x2 == x1:
                    continue
                ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if abs(ang) <= self.config.deskew_max_angle:
                    angles.append(ang)
        if len(angles) >= 5:
            return float(np.median(angles))

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        return float(angle)
