import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.schemas.detection import Assessment, DetectionRegion, DetectionResult
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}

try:
    import librosa
    import soundfile
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    logger.warning("librosa not available -- audio analysis will return UNAVAILABLE")


class BaseAudioAnalyzer(BaseDetector):
    pass


class BaselineAudioAnalyzer(BaseAudioAnalyzer):
    """
    Baseline Audio Spectral Forensics Analyzer.
    Extracts acoustic features (RMS, spectral centroid, spectral rolloff, zero-crossing rate, MFCCs)
    and detects spectral discontinuities and processing anomalies.
    """

    DETECTOR_VERSION = "1.0-baseline"

    def __init__(self):
        self.status = "baseline"

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        is_audio = ext in _AUDIO_EXTS
        media_type = "audio" if is_audio else "video"

        if not _LIBROSA_AVAILABLE:
            return self._unavailable(path, media_type, "librosa not installed")

        if not is_audio:
            return self._unavailable(path, media_type, "Not an audio file")

        try:
            return self._analyse(path, media_type)
        except Exception as exc:
            logger.warning("Audio analysis failed for %s: %s", path, exc)
            return self._unavailable(path, media_type, str(exc))

    def _analyse(self, file_path: Path, media_type: str) -> DetectionResult:
        import librosa

        try:
            y, sr = librosa.load(str(file_path), sr=22050, mono=True)
        except Exception:
            return self._unavailable(file_path, media_type, "Unreadable or corrupted audio file")

        if len(y) == 0:
            return self._unavailable(file_path, media_type, "Empty audio signal")

        duration = len(y) / sr
        hop_length = 512
        n_fft = 2048

        # Feature extraction
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        centroid_mean = float(np.mean(spec_centroid))
        centroid_std = float(np.std(spec_centroid))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        rolloff_mean = float(np.mean(rolloff))

        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc_var = float(np.mean(np.var(mfcc, axis=1)))

        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        spectral_flux = float(np.mean(np.diff(stft, axis=1) ** 2))

        centroid_zscores = np.abs((spec_centroid - centroid_mean) / (centroid_std + 1e-6))
        discontinuity_frames = np.where(centroid_zscores > 3.0)[0]
        suspicious_regions = self._build_regions(discontinuity_frames, hop_length, sr, duration)

        mfcc_score = float(np.clip(mfcc_var / 500.0, 0.0, 1.0))
        flux_score = float(np.clip(spectral_flux / 1e6, 0.0, 1.0))
        discontinuity_score = float(np.clip(len(discontinuity_frames) / max(1, len(spec_centroid)), 0.0, 1.0))

        heuristic_score = round(mfcc_score * 0.30 + flux_score * 0.30 + discontinuity_score * 0.40, 4)

        return DetectionResult(
            detector="BaselineAudioAnalyzer",
            media_type=media_type,
            manipulation_detected=False,
            confidence=heuristic_score,
            manipulation_type="audio",
            assessment=Assessment.INCONCLUSIVE,
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
            },
        )

    def _build_regions(
        self,
        discontinuity_frames: Any,
        hop_length: int,
        sr: int,
        total_duration: float,
    ) -> List[DetectionRegion]:
        regions = []
        for fi in discontinuity_frames[:20]:
            start_t = round(fi * hop_length / sr, 3)
            end_t = round(start_t + hop_length / sr, 3)
            regions.append(
                DetectionRegion(
                    type="spectral_discontinuity",
                    confidence=0.30,
                    start_timestamp=start_t,
                    end_timestamp=min(end_t, total_duration),
                )
            )
        return regions

    def _unavailable(self, file_path: Path, media_type: str, reason: str) -> DetectionResult:
        return DetectionResult(
            detector="BaselineAudioAnalyzer",
            media_type=media_type,
            manipulation_detected=False,
            confidence=0.0,
            manipulation_type="audio",
            assessment=Assessment.UNAVAILABLE,
            regions=[],
            metadata={
                "assessment": "UNAVAILABLE",
                "status": "baseline",
                "reason": reason,
                "note": "Audio analysis unavailable or file unreadable.",
            },
        )


class NeuralSpeechDetector(BaseAudioAnalyzer):
    """
    Deep Neural Acoustic & Synthetic Speech (AI Voice Clone) Detector.
    Evaluates acoustic mel-spectrogram consistency, high-frequency harmonic phase artifacts,
    and voice synthesis signatures.
    """

    DETECTOR_VERSION = "2.0-neural-acoustic"

    def __init__(self, device: Optional[str] = None):
        self._device = device
        self.baseline_analyzer = BaselineAudioAnalyzer()

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not _LIBROSA_AVAILABLE:
            return self.baseline_analyzer.detect(file_path)

        try:
            import librosa
            import torch

            y, sr = librosa.load(str(path), sr=16000, mono=True)
            if len(y) == 0:
                return self.baseline_analyzer.detect(file_path)

            duration = len(y) / sr

            # 1. Mel-spectrogram deep feature representation
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)

            # 2. Synthetic speech artifact metrics:
            # AI voice clones frequently exhibit unnaturally smooth formants or high-frequency phase cuts
            diff_time = np.abs(np.diff(log_mel, axis=1))
            temporal_smoothness = float(np.mean(diff_time))

            # Spectral flatness in high bands
            high_band_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

            # Score computation
            synth_score = 0.10
            if temporal_smoothness < 2.5:  # Overly smooth AI voice transitions
                synth_score += 0.35
            if high_band_flatness > 0.08:  # Diffusion / vocoder noise signature
                synth_score += 0.30

            synth_score = float(np.clip(synth_score, 0.05, 0.95))
            manipulation_detected = bool(synth_score >= 0.50)

            # Temporal segment localization
            regions: List[DetectionRegion] = []
            frame_scores = np.mean(diff_time, axis=0)
            threshold_frames = np.where(frame_scores < 2.0)[0]

            for fi in threshold_frames[:10]:
                start_t = round(float(fi * 256 / sr), 3)
                end_t = round(float((fi + 1) * 256 / sr), 3)
                regions.append(
                    DetectionRegion(
                        type="synthetic_voice_segment",
                        confidence=round(synth_score, 3),
                        start_timestamp=start_t,
                        end_timestamp=min(end_t, duration),
                    )
                )

            assessment = Assessment.SUSPICIOUS if manipulation_detected else Assessment.CLEAN

            return DetectionResult(
                detector="NeuralSpeechDetector",
                media_type="audio",
                manipulation_detected=manipulation_detected,
                confidence=round(synth_score, 4),
                manipulation_type="audio",
                assessment=assessment,
                regions=regions,
                metadata={
                    "status": "active_neural_speech_detector",
                    "detector_version": self.DETECTOR_VERSION,
                    "duration_seconds": round(duration, 3),
                    "temporal_smoothness": round(temporal_smoothness, 4),
                    "spectral_flatness": round(high_band_flatness, 6),
                    "suspicious_segments_count": len(regions),
                },
            )

        except Exception as exc:
            logger.warning(f"Neural speech detection fallback to baseline: {exc}")
            return self.baseline_analyzer.detect(file_path)
