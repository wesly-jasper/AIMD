"""Audio utilities for forensic analysis."""
from pathlib import Path
from typing import Dict, Any

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: Path, sr: int | None = None):
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y, sr_out


def compute_spectrogram(y: np.ndarray, sr: int):
    S = np.abs(librosa.stft(y))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db


def compute_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def audio_features(path: Path) -> Dict[str, Any]:
    y, sr = load_audio(path)
    duration = librosa.get_duration(y=y, sr=sr)
    # Spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    return {
        "sample_rate": sr,
        "duration_seconds": float(duration),
        "spectral_centroid_mean": float(np.mean(cent)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "channels": 1,
    }


def save_waveform_plot(y: np.ndarray, sr: int, out_path: Path) -> Path:
    """Save waveform as image for frontend visualization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, color="#38bdf8")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    return out_path
