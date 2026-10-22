import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from app.schemas.detection import (
    DetectionRegion,
    DetectionResult,
    DetectorAvailability,
    DetectorExecutionStatus,
    ScoreSemantics,
)
from app.services.detection.detectors.base_detector import BaseDetector

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}

try:
    import librosa
    import soundfile
    _LIBROSA_AVAILABLE = True
    _LIBROSA_IMPORT_ERROR: Optional[str] = None
except ImportError as _exc:
    _LIBROSA_AVAILABLE = False
    _LIBROSA_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"
    logger.warning(
        "librosa/soundfile not available -- audio detectors will report "
        "UNAVAILABLE (%s)", _LIBROSA_IMPORT_ERROR
    )


class BaseAudioAnalyzer(BaseDetector):
    manipulation_type = "audio"
    supported_media_types = frozenset({"audio"})

    def availability(self) -> tuple[DetectorAvailability, Optional[str]]:
        """Audio analysis here is entirely librosa/soundfile based."""
        if not _LIBROSA_AVAILABLE:
            return (
                DetectorAvailability.UNAVAILABLE,
                f"librosa/soundfile not installed ({_LIBROSA_IMPORT_ERROR})",
            )
        return DetectorAvailability.AVAILABLE, None


class BaselineAudioAnalyzer(BaseAudioAnalyzer):
    """
    Baseline Audio Spectral Forensics Analyzer.

    Extracts acoustic features (RMS, spectral centroid, spectral rolloff,
    zero-crossing rate, MFCCs) and locates spectral discontinuities.

    A completed run is INCONCLUSIVE by design: these features describe the
    recording's acoustic character, and no threshold has been established that
    separates edited speech from speech that was merely recorded in an unusual
    room. The discontinuity regions are worth an examiner's attention; the
    composite number is not a finding.
    """

    detector_id = "BaselineAudioAnalyzer"
    version = "1.0-baseline"

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        is_audio = ext in _AUDIO_EXTS
        media_type = "audio" if is_audio else "video"

        availability, reason = self.availability()
        if availability is not DetectorAvailability.AVAILABLE:
            return self.unavailable(
                reason or "librosa is unavailable", media_type=media_type
            )

        if not is_audio:
            # AVAILABLE + SKIPPED: this detector works, it was simply not pointed
            # at audio. Recording that as UNAVAILABLE described our own routing
            # decision as a missing dependency.
            return self.skipped(
                f"Not an audio file (extension {ext or 'none'!r} is not one of "
                f"{sorted(_AUDIO_EXTS)})",
                media_type=media_type,
            )

        try:
            return self._analyse(path, media_type)
        except Exception as exc:
            logger.warning("Audio analysis failed for %s: %s", path, exc)
            # AVAILABLE + FAILED. librosa is installed, so the capability is
            # present; this run broke. The two used to be the same label, which
            # made a crashing detector indistinguishable from an uninstalled one.
            return self.failed(
                f"Audio analysis failed: {type(exc).__name__}", media_type=media_type
            )

    def _analyse(self, file_path: Path, media_type: str) -> DetectionResult:
        import librosa

        try:
            y, sr = librosa.load(str(file_path), sr=22050, mono=True)
        except Exception as exc:
            # A problem with the evidence, not with the capability.
            return self.failed(
                f"Unreadable or corrupted audio file ({type(exc).__name__})",
                media_type=media_type,
            )

        if len(y) == 0:
            # Nothing was examined, so there is no observation in either
            # direction. Not a negative result.
            return self.failed(
                "Audio stream decoded to an empty signal; no samples to analyse",
                media_type=media_type,
            )

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

        # Hand-chosen weights over three normalisers that were also hand-chosen
        # (/500, /1e6). No fitting against labelled audio has been performed, so
        # this orders recordings and is not a probability of manipulation.
        heuristic_score = round(mfcc_score * 0.30 + flux_score * 0.30 + discontinuity_score * 0.40, 4)

        # INCONCLUSIVE, so the per-run score is dropped and the regions are kept.
        # That is deliberate: the number has no established threshold, and
        # reporting it as a confidence alongside a COMPLETED status would invite
        # exactly the reading it cannot support.
        return self.inconclusive(
            "Acoustic features extracted and "
            f"{len(discontinuity_frames)} spectral discontinuity frame(s) located. "
            "No validated threshold separates edited audio from unusual but "
            "unedited recordings, so these features localise regions for review "
            "and do not support a determination.",
            media_type=media_type,
            regions=suspicious_regions[:10],
            metadata={
                "detector_version": self.version,
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
                # Reported under its own name, not as a confidence, with its
                # provenance attached.
                "heuristic_score": heuristic_score,
                "heuristic_score_provenance": (
                    "hand-chosen weights over hand-chosen normalisers; not "
                    "calibrated against labelled audio"
                ),
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


class SpectralSpeechHeuristicDetector(BaseAudioAnalyzer):
    """
    Synthetic-speech *heuristic* over mel-spectrogram statistics.

    WHY THIS CLASS WAS RENAMED
    --------------------------
    It was called ``NeuralSpeechDetector``, documented as a "Deep Neural Acoustic
    & Synthetic Speech (AI Voice Clone) Detector", and versioned
    "2.0-neural-acoustic". It performs **no neural inference**. It imported
    ``torch`` and never used it, then computed two librosa statistics --
    mean absolute frame-to-frame difference of the log-mel spectrogram, and mean
    spectral flatness -- and compared each to a hand-picked constant.

    The name, the docstring and the version string were therefore all claims
    about a capability that does not exist in this repository, and they were the
    text an examiner would read when asking what produced a result. The class is
    renamed, the version string says "heuristic", and the unused ``torch`` import
    is gone.

    What it actually measures: AI voice synthesis *can* leave over-smooth formant
    transitions and vocoder noise in the high bands. So can low-bitrate
    compression, noise suppression, and a close-miked studio recording. The two
    thresholds below were chosen by judgement against no labelled corpus, so a
    completed run reports HEURISTIC_SEVERITY and localises segments; it is not a
    voice-clone determination.
    """

    detector_id = "SpectralSpeechHeuristicDetector"
    version = "2.0-spectral-heuristic"

    def __init__(self, device: Optional[str] = None):
        self._device = device

    def detect(self, file_path: str) -> DetectionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        availability, reason = self.availability()
        if availability is not DetectorAvailability.AVAILABLE:
            # Previously every one of these paths returned
            # ``self.baseline_analyzer.detect(file_path)``: a different
            # detector's result, under a different detector's identity, with
            # nothing recording that this detector had been requested. Three
            # separate situations -- librosa missing, an empty signal, and a
            # crash -- all arrived as a BaselineAudioAnalyzer result.
            return self.unavailable(reason or "librosa is unavailable")

        try:
            import librosa

            y, sr = librosa.load(str(path), sr=16000, mono=True)
        except Exception as exc:
            return self.failed(
                f"Audio could not be decoded for spectral analysis "
                f"({type(exc).__name__})"
            )

        if len(y) == 0:
            return self.failed(
                "Audio stream decoded to an empty signal; no samples to analyse"
            )

        try:
            duration = len(y) / sr

            # 1. Mel-spectrogram representation (a signal transform, not a model).
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=256)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)

            # 2. Two hand-thresholded statistics. Synthetic speech often shows
            # over-smooth frame-to-frame transitions and flat high bands -- and so
            # does heavily compressed or noise-suppressed natural speech, which is
            # why neither threshold can carry a conclusion on its own.
            diff_time = np.abs(np.diff(log_mel, axis=1))
            temporal_smoothness = float(np.mean(diff_time))

            high_band_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

            synth_score = 0.10
            if temporal_smoothness < 2.5:  # over-smooth transitions
                synth_score += 0.35
            if high_band_flatness > 0.08:  # vocoder / codec noise signature
                synth_score += 0.30

            synth_score = float(np.clip(synth_score, 0.05, 0.95))
            anomaly_flagged = bool(synth_score >= 0.50)

            # Temporal segment localization
            regions: List[DetectionRegion] = []
            frame_scores = np.mean(diff_time, axis=0)
            threshold_frames = np.where(frame_scores < 2.0)[0]

            for fi in threshold_frames[:10]:
                start_t = round(float(fi * 256 / sr), 3)
                end_t = round(float((fi + 1) * 256 / sr), 3)
                regions.append(
                    DetectionRegion(
                        type="over_smooth_spectral_segment",
                        confidence=round(synth_score, 3),
                        start_timestamp=start_t,
                        end_timestamp=min(end_t, duration),
                    )
                )
        except Exception as exc:
            logger.warning("Spectral speech heuristic failed: %s", exc)
            return self.failed(
                f"Spectral speech analysis failed ({type(exc).__name__})"
            )

        return DetectionResult(
            detector=self.detector_id,
            detector_version=self.version,
            media_type="audio",
            manipulation_detected=anomaly_flagged,
            confidence=round(synth_score, 4),
            manipulation_type=self.manipulation_type,
            availability=DetectorAvailability.AVAILABLE,
            execution_status=DetectorExecutionStatus.COMPLETED,
            # Two hand-picked thresholds over two librosa statistics. The old code
            # assigned ``Assessment.CLEAN`` below the threshold, which let a
            # heuristic with no validated operating point report an audio file as
            # clean. The status fields now carry that, and CLEAN is derived only
            # from COMPLETED + no anomaly -- still not a finding of authenticity.
            score_semantics=ScoreSemantics.HEURISTIC_SEVERITY,
            regions=regions,
            metadata={
                "detector_version": self.version,
                "performs_neural_inference": False,
                "method": (
                    "log-mel temporal smoothness and spectral flatness, "
                    "hand-thresholded"
                ),
                "duration_seconds": round(duration, 3),
                "temporal_smoothness": round(temporal_smoothness, 4),
                "spectral_flatness": round(high_band_flatness, 6),
                "suspicious_segments_count": len(regions),
            },
        )
