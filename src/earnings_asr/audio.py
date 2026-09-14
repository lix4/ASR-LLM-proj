"""Bounded, silence-aware segmentation with sample-exact, nonoverlapping coverage."""

import math
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16000


def load_audio(path, max_seconds=None):
    info = sf.info(path)
    if not info.frames or info.samplerate <= 0:
        raise ValueError("Audio is empty or has invalid metadata")
    if max_seconds is not None and info.duration > max_seconds:
        raise ValueError(f"Audio exceeds {max_seconds} seconds")
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        factor = math.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // factor, sr // factor).astype(np.float32)
    return audio


@dataclass
class Chunk:
    start: int
    end: int
    silent: bool


def segment(audio, chunk_seconds=25.0, search_seconds=2.0, silence_rms=0.0001):
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive and finite")
    if not 0 <= search_seconds < chunk_seconds or silence_rms < 0:
        raise ValueError("Invalid boundary search or silence threshold")
    maximum = max(1, round(chunk_seconds * SAMPLE_RATE))
    search = round(search_seconds * SAMPLE_RATE)
    frame = 320  # 20 ms; boundary timestamps are segment boundaries, not word alignments.
    start = 0
    while start < len(audio):
        end = min(start + maximum, len(audio))
        if end < len(audio) and search >= frame:
            lo = max(start + frame, end - search)
            candidates = np.arange(lo, end - frame + 1, frame)
            if len(candidates):
                energies = [float(np.mean(audio[p:p + frame] ** 2)) for p in candidates]
                # Only move toward an actual quiet interval; never overlap windows.
                quietest = int(np.argmin(energies))
                if energies[quietest] < 1e-4:
                    end = int(candidates[quietest]) + frame // 2
        piece = audio[start:end]
        silent = float(np.sqrt(np.mean(piece.astype(np.float64) ** 2))) <= silence_rms
        yield Chunk(start, end, silent)
        start = end
