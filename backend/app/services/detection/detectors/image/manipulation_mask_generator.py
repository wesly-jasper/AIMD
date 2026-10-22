"""
Manipulation Mask Generator for Pixel-Level Localization.

Generates high-resolution manipulation heatmaps and segmentation masks
using multi-scale analysis, frequency domain features, and deep learning.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image

from app.schemas.detection import (
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)


class ManipulationMaskGenerator(BaseDetector):
    """
    Generates pixel-level manipulation masks using multi-scale analysis.

    Features:
    - Multi-scale ELA (Error Level Analysis) at different quality factors
    - Multi-scale 2D-FFT frequency analysis
    - Noise residual analysis at multiple scales
    - Gradient-based edge consistency
    - Heatmap fusion with adaptive weighting
    - Optional: UNet-style segmentation (if model available)
    """

    def __init__(
        self,
        scales: List[int] = None,  # Analysis scales (downsample factors)
        quality_factors: List[int] = None,  # ELA quality factors
        gaussian_sigma: float = 1.5,
        heatmap_threshold: float = 0.3,
        min_region_area: int = 64,  # Minimum region area in pixels
        enable_unet: bool = False,  # Enable deep learning segmentation
        unet_model_path: Optional[str] = None,
    ):
        self.scales = scales or [1, 2, 4, 8]
        self.quality_factors = quality_factors or [75, 85, 95]
        self.gaussian_sigma = gaussian_sigma
        self.heatmap_threshold = heatmap_threshold
        self.min_region_area = min_region_area
        self.enable_unet = enable_unet
        self.unet_model_path = unet_model_path
        self._unet_model = None

    detector_id = "ManipulationMaskGenerator"
    version = "1.0.0"
    supported_media_types = frozenset({"image"})
    manipulation_type = "manipulation"

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            # The detector works; the evidence was not readable. Reporting this as
            # UNAVAILABLE would blame the capability for a missing input.
            return self.failed("Image input file is missing")

        try:
            img = cv2.imread(str(path))
            if img is None:
                return self.failed("Image input is empty, invalid, or unsupported")

            h, w = img.shape[:2]

            # Generate multi-scale heatmap
            heatmap = self._generate_multiscale_heatmap(img)

            # Smooth and normalize
            heatmap = self._postprocess_heatmap(heatmap, (w, h))

            # Extract regions from heatmap
            regions = self._extract_regions_from_heatmap(heatmap, (w, h))

            # Compute overall confidence
            max_conf = max((r.confidence for r in regions), default=0.0)
            manipulation_detected = max_conf >= 0.45 and len(regions) > 0

            # If UNet enabled and model available, refine with deep learning
            if self.enable_unet and self._unet_model is not None:
                regions = self._refine_with_unet(img, heatmap, regions)

            return DetectionResult(
                detector=self.detector_id,
                detector_version=self.version,
                media_type="image",
                manipulation_detected=manipulation_detected,
                confidence=round(max_conf, 4),
                manipulation_type=self.manipulation_type,
                availability=DetectorAvailability.AVAILABLE,
                execution_status=DetectorExecutionStatus.COMPLETED,
                # A fused multi-scale ELA/FFT/noise heatmap maximum. Ordinal
                # severity only -- it is not a probability of manipulation.
                score_semantics=ScoreSemantics.HEURISTIC_SEVERITY,
                regions=regions,
                metadata={
                    "status": "active_mask_generation",
                    "scales_used": self.scales,
                    "quality_factors": self.quality_factors,
                    "heatmap_shape": list(heatmap.shape),
                    "heatmap_stats": {
                        "min": float(heatmap.min()),
                        "max": float(heatmap.max()),
                        "mean": float(heatmap.mean()),
                        "std": float(heatmap.std()),
                    },
                    "regions_found": len(regions),
                    "unet_refinement": (
                        "unavailable" if self.enable_unet and self._unet_model is None
                        else "configured" if self.enable_unet
                        else "disabled"
                    ),
                    "note": "OBSERVATION: Pixel-level manipulation heatmap generated. Regions indicate suspicious areas.",
                },
            )

        except Exception as e:
            logger.warning(f"ManipulationMaskGenerator failed: {e}")
            # AVAILABLE + FAILED. This was previously reported as UNAVAILABLE,
            # which made a crash in mask computation look like a detector that had
            # never been installed -- hiding a defect that needs a code fix behind
            # what reads as a configuration gap.
            return self.failed(f"Mask computation failed: {type(e).__name__}")

    def _generate_multiscale_heatmap(self, img: np.ndarray) -> np.ndarray:
        """Generate manipulation heatmap at multiple scales and fuse them."""
        h, w = img.shape[:2]
        fused_heatmap = np.zeros((h, w), dtype=np.float32)

        for scale in self.scales:
            if scale == 1:
                scaled_img = img
            else:
                scaled_h, scaled_w = h // scale, w // scale
                scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

            # Generate heatmap at this scale
            scale_heatmap = self._generate_single_scale_heatmap(scaled_img)

            # Upsample back to original size
            if scale != 1:
                scale_heatmap = cv2.resize(scale_heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

            # Weight by scale (finer scales get higher weight for detail)
            weight = 1.0 / scale
            fused_heatmap += scale_heatmap * weight

        # Normalize by sum of weights
        total_weight = sum(1.0 / s for s in self.scales)
        fused_heatmap /= total_weight

        return fused_heatmap

    def _generate_single_scale_heatmap(self, img: np.ndarray) -> np.ndarray:
        """Generate heatmap at a single scale using multiple cues."""
        h, w = img.shape[:2]
        heatmaps = []

        # 1. Multi-quality ELA
        for qf in self.quality_factors:
            ela = self._compute_ela(img, qf)
            heatmaps.append(ela)

        # 2. Multi-scale FFT analysis
        fft_heatmap = self._compute_fft_heatmap(img)
        heatmaps.append(fft_heatmap)

        # 3. Noise residual (PRNU-like)
        noise_heatmap = self._compute_noise_heatmap(img)
        heatmaps.append(noise_heatmap)

        # 4. Gradient consistency
        grad_heatmap = self._compute_gradient_heatmap(img)
        heatmaps.append(grad_heatmap)

        # 5. Color space inconsistency (RGB vs LAB)
        color_heatmap = self._compute_color_heatmap(img)
        heatmaps.append(color_heatmap)

        # Fuse with adaptive weights
        weights = [1.0, 1.0, 1.0, 0.8, 1.0, 0.7]  # Adjust based on importance
        fused = np.zeros_like(heatmaps[0], dtype=np.float32)
        for hm, w in zip(heatmaps, weights):
            fused += hm * w
        fused /= sum(weights)

        return fused

    def _compute_ela(self, img: np.ndarray, quality: int) -> np.ndarray:
        """Compute Error Level Analysis at given quality factor."""
        # Save as JPEG at quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', img, encode_param)
        recompressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if recompressed is None:
            return np.zeros(img.shape[:2], dtype=np.float32)

        # Compute difference
        diff = cv2.absdiff(img.astype(np.float32), recompressed.astype(np.float32))
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # Normalize
        ela = diff_gray.astype(np.float32) / 255.0

        # Apply Gaussian smoothing
        ela = cv2.GaussianBlur(ela, (0, 0), self.gaussian_sigma)

        return ela

    def _compute_fft_heatmap(self, img: np.ndarray) -> np.ndarray:
        """Compute frequency domain anomaly heatmap using sliding window FFT."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape

        # Use overlapping windows
        win_size = 64
        stride = 32
        heatmap = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - win_size + 1, stride):
            for x in range(0, w - win_size + 1, stride):
                patch = gray[y:y+win_size, x:x+win_size]

                # Apply window function
                window = np.hanning(win_size)
                window_2d = np.outer(window, window)
                patch = patch * window_2d

                # Compute FFT
                f = np.fft.fft2(patch)
                fshift = np.fft.fftshift(f)
                magnitude = np.log(np.abs(fshift) + 1e-10)

                # High frequency energy (anomaly indicator)
                cy, cx = win_size // 2, win_size // 2
                # Mask out low frequencies
                y_grid, x_grid = np.ogrid[:win_size, :win_size]
                dist = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
                hf_mask = dist > win_size * 0.15

                hf_energy = np.mean(magnitude[hf_mask])
                total_energy = np.mean(magnitude)
                anomaly_score = hf_energy / (total_energy + 1e-10)

                heatmap[y:y+win_size, x:x+win_size] += anomaly_score
                count[y:y+win_size, x:x+win_size] += 1

        # Normalize by count
        count[count == 0] = 1
        heatmap /= count

        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)

        return heatmap

    def _compute_noise_heatmap(self, img: np.ndarray) -> np.ndarray:
        """Compute noise residual heatmap (PRNU-like)."""
        # Use wavelet-based noise extraction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Simple high-pass filter for noise
        blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        noise = gray - blur

        # Local variance of noise
        kernel_size = 16
        noise_var = cv2.boxFilter(noise**2, -1, (kernel_size, kernel_size))
        mean_noise = cv2.boxFilter(noise, -1, (kernel_size, kernel_size))
        local_var = noise_var - mean_noise**2

        # Normalize
        heatmap = cv2.GaussianBlur(local_var, (0, 0), self.gaussian_sigma)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)

        return heatmap

    def _compute_gradient_heatmap(self, img: np.ndarray) -> np.ndarray:
        """Compute gradient inconsistency heatmap."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape

        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)

        # Local gradient coherence
        win_size = 16
        coherence = np.zeros_like(mag)

        for y in range(0, h - win_size + 1, win_size // 2):
            for x in range(0, w - win_size + 1, win_size // 2):
                patch_angle = angle[y:y+win_size, x:x+win_size]
                # Circular variance
                mean_cos = np.mean(np.cos(2 * patch_angle))
                mean_sin = np.mean(np.sin(2 * patch_angle))
                coh = np.sqrt(mean_cos**2 + mean_sin**2)
                coherence[y:y+win_size, x:x+win_size] = coh

        # Incoherence = anomaly
        heatmap = 1.0 - coherence
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), self.gaussian_sigma)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)

        return heatmap

    def _compute_color_heatmap(self, img: np.ndarray) -> np.ndarray:
        """Compute color space inconsistency heatmap."""
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        h, w = l.shape

        # Check for unnatural color transitions
        # Compute local statistics in LAB space
        win_size = 32
        heatmap = np.zeros_like(l)

        for y in range(0, h - win_size + 1, win_size // 2):
            for x in range(0, w - win_size + 1, win_size // 2):
                patch_l = l[y:y+win_size, x:x+win_size]
                patch_a = a[y:y+win_size, x:x+win_size]
                patch_b = b[y:y+win_size, x:x+win_size]

                # Check if chroma channels are constant (synthetic)
                std_a = np.std(patch_a)
                std_b = np.std(patch_b)
                # Low chroma variation may indicate manipulation
                anomaly = 1.0 / (1.0 + std_a + std_b)

                heatmap[y:y+win_size, x:x+win_size] += anomaly

        # Normalize
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), self.gaussian_sigma)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)

        return heatmap

    def _postprocess_heatmap(self, heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Post-process heatmap: resize, smooth, threshold."""
        w, h = target_size

        # Resize if needed
        if heatmap.shape[1] != w or heatmap.shape[0] != h:
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

        # Gaussian smoothing
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), self.gaussian_sigma)

        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)

        return heatmap

    def _extract_regions_from_heatmap(
        self, heatmap: np.ndarray, img_size: Tuple[int, int]
    ) -> List[DetectionRegion]:
        """Extract connected regions from heatmap above threshold."""
        w, h = img_size

        # Threshold
        binary = (heatmap > self.heatmap_threshold).astype(np.uint8)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_region_area:
                continue

            x, y, w_rect, h_rect = cv2.boundingRect(cnt)

            # Compute mean confidence in region
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            region_confidence = float(np.mean(heatmap[mask > 0]))

            # Only keep high-confidence regions
            if region_confidence < 0.35:
                continue

            regions.append(DetectionRegion(
                type="manipulation_mask",
                confidence=round(region_confidence, 3),
                bbox=[float(x), float(y), float(w_rect), float(h_rect)],
                extra={
                    "area": float(area),
                    "mean_heatmap_value": round(region_confidence, 3),
                    "contour_points": len(cnt),
                }
            ))

        # Sort by confidence
        regions.sort(key=lambda r: r.confidence, reverse=True)
        return regions[:15]

    def _refine_with_unet(
        self, img: np.ndarray, heatmap: np.ndarray, regions: List[DetectionRegion]
    ) -> List[DetectionRegion]:
        """Refine regions using UNet segmentation (placeholder for future integration)."""
        # TODO: Load and run UNet model for precise segmentation
        # This would replace bounding boxes with pixel-perfect masks
        return regions

    # `_inconclusive` and `_unavailable` were removed here. Both are now on
    # BaseDetector as `inconclusive()`, `unavailable()`, `failed()` and
    # `skipped()`, so the four situations cannot be conflated and every detector
    # reports them the same way. The old local `_unavailable` was being used for
    # a crash, which is `failed()`.


class DeepfakeSegmentationModel:
    """
    Placeholder for future UNet/DeepLab integration for pixel-perfect segmentation.
    Would load a pre-trained model for manipulation localization.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self):
        """Load the segmentation model (PyTorch/ONNX)."""
        pass

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Predict manipulation mask."""
        pass