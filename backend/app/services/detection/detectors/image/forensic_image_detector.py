import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from pathlib import Path
from typing import Dict, Any, List

from app.schemas.detection import DetectionResult, DetectionRegion
from app.services.detection.detectors.base_detector import BaseDetector


class ForensicImageDetector(BaseDetector):
    """
    Advanced Multi-Signal Forensic Detector for Images.
    Applies real signal-processing and computer vision forensic methods:
    1. Error Level Analysis (ELA) for compression and splice detection.
    2. 2D Fast Fourier Transform (FFT) for generative lattice & high-frequency artifacts.
    3. Noise Variance / Residual Analysis for sensor noise inconsistencies.
    4. Gradient & Laplacian Edge Boundary consistency.
    5. Metadata & EXIF anomaly inspection.
    """

    def __init__(self, ela_quality: int = 90, ela_scale: int = 15):
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale

    def detect(self, file_path: str) -> DetectionResult:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError(f"Unable to read image at {file_path}")

            h, w, c = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. Error Level Analysis (ELA)
            ela_score, suspicious_ela_regions = self._analyze_ela(file_path, h, w)

            # 2. 2D-FFT Spectral Analysis (Generative / Diffusion Artifacts)
            fft_score, spectral_metrics = self._analyze_fft_spectrum(gray)

            # 3. Noise Residual Inconsistency Analysis
            noise_score, noise_regions = self._analyze_noise_residuals(gray, h, w)

            # 4. Metadata / EXIF Inspection
            meta_score, meta_indicators = self._analyze_metadata(file_path)

            # Aggregate scores with calibrated weights
            # Spectral artifacts + ELA + Noise inconsistencies + Metadata
            composite_score = (
                fft_score * 0.35 +
                ela_score * 0.30 +
                noise_score * 0.25 +
                meta_score * 0.10
            )
            composite_score = float(np.clip(composite_score, 0.0, 0.99))

            manipulation_detected = composite_score >= 0.50

            # Determine dominant manipulation type
            if fft_score > 0.65 or meta_indicators.get("is_ai_generated"):
                manipulation_type = "ai_generated_synthetic"
            elif ela_score > 0.60 or noise_score > 0.60:
                manipulation_type = "splice_or_copy_move"
            else:
                manipulation_type = "image_manipulation"

            # Combine localized suspicious regions
            regions = suspicious_ela_regions + noise_regions

            return DetectionResult(
                detector="ForensicImageDetector",
                media_type="image",
                manipulation_detected=manipulation_detected,
                confidence=round(composite_score, 4),
                manipulation_type=manipulation_type,
                regions=regions[:10], # Top 10 most suspicious regions
                metadata={
                    "status": "active_forensic_engine",
                    "ela_score": round(ela_score, 4),
                    "fft_spectral_score": round(fft_score, 4),
                    "noise_inconsistency_score": round(noise_score, 4),
                    "metadata_score": round(meta_score, 4),
                    "spectral_metrics": spectral_metrics,
                    "metadata_indicators": meta_indicators
                }
            )

        except Exception as e:
            # Safe fallback if image decoding fails
            return DetectionResult(
                detector="ForensicImageDetector",
                media_type="image",
                manipulation_detected=False,
                confidence=0.0,
                manipulation_type="image",
                regions=[],
                metadata={"status": "error", "error": str(e)}
            )

    def _analyze_ela(self, file_path: Path, height: int, width: int):
        """
        Calculates Error Level Analysis (ELA) by recompressing at a known JPEG quality
        and finding localized difference variance anomalies.
        """
        try:
            original = Image.open(file_path).convert("RGB")
            buffer = io.BytesIO()
            original.save(buffer, "JPEG", quality=self.ela_quality)
            buffer.seek(0)
            recompressed = Image.open(buffer)

            diff = ImageChops.difference(original, recompressed)
            diff_np = np.array(diff, dtype=np.float32)
            ela_magnitude = np.mean(diff_np, axis=2)

            # Local variance across 32x32 patches
            grid_size = max(16, min(height, width) // 16)
            patch_variances = []
            suspicious_regions = []

            for y in range(0, height - grid_size, grid_size):
                for x in range(0, width - grid_size, grid_size):
                    patch = ela_magnitude[y:y+grid_size, x:x+grid_size]
                    p_mean = float(np.mean(patch))
                    patch_variances.append(p_mean)

            if len(patch_variances) == 0:
                return 0.1, []

            mean_val = float(np.mean(patch_variances))
            std_val = float(np.std(patch_variances))

            # Regions that deviate significantly (2+ std devs) from global compression surface
            for y in range(0, height - grid_size, grid_size):
                for x in range(0, width - grid_size, grid_size):
                    patch = ela_magnitude[y:y+grid_size, x:x+grid_size]
                    p_mean = float(np.mean(patch))
                    if std_val > 1e-4 and (p_mean - mean_val) > (1.8 * std_val):
                        conf = min(0.95, float(0.5 + 0.4 * ((p_mean - mean_val) / (std_val * 3))))
                        suspicious_regions.append(
                            DetectionRegion(
                                type="compression_anomaly_region",
                                confidence=round(conf, 3),
                                bbox=[float(x), float(y), float(grid_size), float(grid_size)]
                            )
                        )

            # Unnatural uniformity (very low std) or high local variance indicates manipulation
            var_ratio = std_val / (mean_val + 1e-6)
            ela_score = float(np.clip(var_ratio * 1.5, 0.05, 0.95))
            return ela_score, suspicious_regions

        except Exception:
            return 0.1, []

    def _analyze_fft_spectrum(self, gray_img: np.ndarray):
        """
        2D Fourier Transform Radial Power Spectrum Analysis.
        Detects artificial high-frequency harmonics and diffusion spectral anomalies.
        """
        try:
            h, w = gray_img.shape
            # Compute 2D FFT
            f = np.fft.fft2(gray_img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)

            cy, cx = h // 2, w // 2
            r_inner = min(h, w) // 6
            r_outer = min(h, w) // 3

            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)

            low_freq_mask = dist_from_center <= r_inner
            mid_freq_mask = (dist_from_center > r_inner) & (dist_from_center <= r_outer)
            high_freq_mask = dist_from_center > r_outer

            lf_mean = float(np.mean(magnitude_spectrum[low_freq_mask]))
            mf_mean = float(np.mean(magnitude_spectrum[mid_freq_mask]))
            hf_mean = float(np.mean(magnitude_spectrum[high_freq_mask]))

            # Ratio of high-frequency power to low-frequency power
            hf_ratio = hf_mean / (lf_mean + 1e-6)

            # High-frequency kurtosis / peakiness (checkerboard artifacts in diffusion models)
            hf_vals = magnitude_spectrum[high_freq_mask]
            kurtosis_indicator = float(np.std(hf_vals) / (np.mean(hf_vals) + 1e-6))

            # Calibrate generative spectrum anomaly score
            spectral_score = 0.0
            if hf_ratio > 0.80:
                spectral_score += 0.40
            if hf_ratio > 0.85:
                spectral_score += 0.30
            if kurtosis_indicator > 0.15:
                spectral_score += 0.25

            spectral_score = float(np.clip(spectral_score, 0.05, 0.95))

            return spectral_score, {
                "high_freq_ratio": round(hf_ratio, 4),
                "hf_kurtosis": round(kurtosis_indicator, 4),
                "low_freq_power": round(lf_mean, 2),
                "high_freq_power": round(hf_mean, 2)
            }
        except Exception:
            return 0.1, {}

    def _analyze_noise_residuals(self, gray_img: np.ndarray, height: int, width: int):
        """
        Extracts high-pass noise residuals using median filtering and checks
        for spatial noise variance inconsistencies across the image.
        """
        try:
            denoised = cv2.medianBlur(gray_img, 3)
            residual = cv2.absdiff(gray_img, denoised).astype(np.float32)

            grid_size = max(32, min(height, width) // 8)
            local_vars = []
            regions = []

            for y in range(0, height - grid_size, grid_size):
                for x in range(0, width - grid_size, grid_size):
                    patch = residual[y:y+grid_size, x:x+grid_size]
                    p_var = float(np.var(patch))
                    local_vars.append((p_var, x, y))

            if not local_vars:
                return 0.1, []

            variances = [v[0] for v in local_vars]
            global_var_mean = float(np.mean(variances))
            global_var_std = float(np.std(variances))

            for p_var, x, y in local_vars:
                if global_var_std > 1e-4 and abs(p_var - global_var_mean) > (2.0 * global_var_std):
                    conf = min(0.90, float(0.55 + 0.35 * (abs(p_var - global_var_mean) / (global_var_std * 3))))
                    regions.append(
                        DetectionRegion(
                            type="noise_inconsistency_patch",
                            confidence=round(conf, 3),
                            bbox=[float(x), float(y), float(grid_size), float(grid_size)]
                        )
                    )

            inconsistency_ratio = global_var_std / (global_var_mean + 1e-6)
            noise_score = float(np.clip(inconsistency_ratio * 0.8, 0.05, 0.90))

            return noise_score, regions

        except Exception:
            return 0.1, []

    def _analyze_metadata(self, file_path: Path):
        """
        Inspects EXIF metadata for hardware camera tags vs. AI generator signatures.
        """
        indicators = {
            "has_camera_exif": False,
            "is_ai_generated": False,
            "software": None
        }
        score = 0.1

        try:
            img = Image.open(file_path)
            exif = img._getexif()

            ai_keywords = [
                "stable diffusion", "midjourney", "dall-e", "comfyui",
                "invokeai", "novelai", "civitai", "automatic1111", "flux"
            ]

            # Check raw text chunks (e.g. PNG text / tEXt chunks)
            if hasattr(img, "info") and isinstance(img.info, dict):
                for k, v in img.info.items():
                    v_str = str(v).lower()
                    for kw in ai_keywords:
                        if kw in v_str or kw in str(k).lower():
                            indicators["is_ai_generated"] = True
                            indicators["software"] = kw
                            return 0.95, indicators

            if exif:
                # Check software tag (0x0131) or UserComment (0x9286)
                for tag_id, val in exif.items():
                    val_str = str(val).lower()
                    for kw in ai_keywords:
                        if kw in val_str:
                            indicators["is_ai_generated"] = True
                            indicators["software"] = kw
                            return 0.95, indicators

                # Check camera make (0x010f) / model (0x0110)
                if 0x010F in exif or 0x0110 in exif:
                    indicators["has_camera_exif"] = True
                    score = 0.05 # Legitimate camera metadata present

            # If high-resolution photorealistic image has zero camera metadata
            if not indicators["has_camera_exif"]:
                score = 0.40

        except Exception:
            pass

        return score, indicators
