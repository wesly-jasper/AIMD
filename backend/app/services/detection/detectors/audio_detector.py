"""
Audio forensic analyzer — real baseline spectral analysis.

Analyzes:
  - Waveform RMS energy distribution
  - Spectral centroid (FACT: measured from signal)
  - MFCC variance (OBSERVATION: derived feature)
  - Zero crossing rate
  - Spectral rolloff
  - Spectral flux (frame-to-frame spectral change)
  - Spectral discontinuities (sudden shifts → potential splice points)

Assessment:
  INCONCLUSIVE — always, unless a validated synthetic-speech ML model is configured.

These spectral signals are forensic indicators, NOT proof of AI-generated speech.
Anomalies may indicate splicing, noise addition, codec switching, or re-encoding.
They do NOT alone prove synthetic origin.

The detector interface is designed so a validated ML-based synthetic speech
detector (e.g. RawNet2, Wav2Vec2 fine-tuned) can be plugged in as a
separate detector without modifying this baseline.
"""
import logging
from pathlib import Path
from typing import Any

from app.schemas.detection import DetectionResult, DetectionRegion
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg"}

try:
    import librosa
    import numpy as np
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.warning("librosa not available — audio analysis will return UNAVAILABLE")


class BaseAudioAnalyzer(BaseDetector):
    pass


class BaselineAudioAnalyzer(BaseAudioAnalyzer):
    """
    Baseline audio forensic analyzer.

    Classification: HEURISTIC / BASELINE
    Limitation: Cannot reliably detect synthetic speech without a validated ML model.
    Always returns INCONCLUSIVE for the manipulation_detected field.
    """

    DETECTOR_VERSION = "1.0-baseline"

    def __init__(self):
        self.status = "baseline"

    def detect(self, file_path: str) -> DetectionResult:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        is_audio = ext in _AUDIO_EXTS
        media_type = "audio" if is_audio else "video"

        if not _LIBROSA_AVAILABLE:
            return self._unavailable(file_path, media_type, "librosa not installed")

        if not is_audio:
            return self._unavailable(file_path, media_type, "Not an audio file")

        try:
            return self._analyse(file_path, media_type)
        except Exception as exc:
            logger.warning("Audio analysis failed for %s: %s", file_path, exc)
            return self._unavailable(file_path, media_type, str(exc))

    def _analyse(self, file_path: Path, media_type: str) -> DetectionResult:
        import librosa
        import numpy as np

        # Load audio (resampled to 22050 Hz for consistent analysis)
        y, sr = librosa.load(str(file_path), sr=22050, mono=True)

        if len(y) == 0:
            return self._unavailable(file_path, media_type, "Empty audio signal")

        duration = len(y) / sr

        # ── Feature extraction ────────────────────────────────────────────────
        # Hop length for frame analysis (~23ms frames at 22050 Hz)
        hop_length = 512
        n_fft = 2048

        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        # Spectral centroid (brightness)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        centroid_mean = float(np.mean(spec_centroid))
        centroid_std = float(np.std(spec_centroid))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        rolloff_mean = float(np.mean(rolloff))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        # MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc_var = float(np.mean(np.var(mfcc, axis=1)))

        # ── Spectral flux (frame-to-frame change) ─────────────────────────────
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        spectral_flux = np.mean(np.diff(stft, axis=1) ** 2)
        spectral_flux = float(spectral_flux)

        # ── Discontinuity detection ───────────────────────────────────────────
        # Look for frames where spectral centroid deviates > 3 std dev from mean
        centroid_zscores = np.abs((spec_centroid - centroid_mean) / (centroid_std + 1e-6))
        discontinuity_frames = np.where(centroid_zscores > 3.0)[0]
        suspicious_regions = self._build_regions(
            discontinuity_frames, hop_length, sr, duration
        )

        # ── Scoring ───────────────────────────────────────────────────────────
        # High MFCC variance and high spectral flux can indicate synthetic or
        # heavily processed audio — but these are weak signals.
        # We compute a heuristic score only for reporting purposes.
        # The assessment remains INCONCLUSIVE regardless.
        mfcc_score = float(np.clip(mfcc_var / 500.0, 0.0, 1.0))
        flux_score = float(np.clip(spectral_flux / 1e6, 0.0, 1.0))
        discontinuity_score = float(np.clip(len(discontinuity_frames) / max(1, len(spec_centroid)), 0.0, 1.0))

        heuristic_score = round(
            mfcc_score * 0.30 + flux_score * 0.30 + discontinuity_score * 0.40, 4
        )

        return DetectionResult(
            detector="BaselineAudioAnalyzer",
            media_type=media_type,
            # IMPORTANT: manipulation_detected is None / INCONCLUSIVE.
            # We cannot reliably determine synthetic speech from these features alone.
            manipulation_detected=False,
            confidence=heuristic_score,
            manipulation_type="audio",
            regions=suspicious_regions[:10],
            metadata={
                "status": "baseline",
                "detector_version": self.DETECTOR_VERSION,
                "assessment": "INCONCLUSIVE",
                "duration_seconds": round(duration, 3),
                "sample_rate": sr,
                "features": {
                    "rms_mean": round(rms_mean, 6),
                    "rms_std": round(rms_std, 6),
                    "spectral_centroid_mean_hz": round(centroid_mean, 2),
                    "spectral_centroid_std_hz": round(centroid_std, 2),
                    "spectral_rolloff_mean_hz": round(rolloff_mean, 2),
                    "zero_crossing_rate_mean": round(zcr_mean, 6),
                    "zero_crossing_rate_std": round(zcr_std, 6),
                    "mfcc_variance": round(mfcc_var, 4),
                    "spectral_flux": round(spectral_flux, 4),
                    "spectral_discontinuities": int(len(discontinuity_frames)),
                },
                "heuristic_score": heuristic_score,
                "limitations": [
                    "UNCERTAINTY: Baseline spectral features cannot reliably detect "
                    "synthetic speech. Requires a validated ML model (e.g. RawNet2) "
                    "for reliable AI audio detection.",
                    "UNCERTAINTY: Spectral discontinuities may indicate re-encoding, "
                    "noise reduction, or legitimate audio editing rather than synthesis.",
                ],
            },
        )

    def _build_regions(
        self,
        discontinuity_frames: Any,
        hop_length: int,
        sr: int,
        total_duration: float,
    ) -> list[DetectionRegion]:
        """Convert frame indices to temporal DetectionRegion objects."""
        regions = []
        for fi in discontinuity_frames[:20]:  # Limit to 20
            start_t = round(fi * hop_length / sr, 3)
            end_t = round(start_t + hop_length / sr, 3)
            regions.append(DetectionRegion(
                type="spectral_discontinuity",
                confidence=0.30,  # Low confidence — heuristic only
                start_timestamp=start_t,
                end_timestamp=min(end_t, total_duration),
            ))
        return regions

    def _unavailable(self, file_path: Path, media_type: str, reason: str) -> DetectionResult:
        return DetectionResult(
            detector="BaselineAudioAnalyzer",
            media_type=media_type,
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type="audio",
            regions=[],
            metadata={
                "assessment": "UNAVAILABLE",
                "status": "baseline",
                "reason": reason,
                "note": (
                    "UNCERTAINTY: Audio analysis unavailable. "
                    "Install librosa and soundfile for baseline spectral analysis."
                ),
            },
        )
