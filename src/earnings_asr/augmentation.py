"""Deterministic building blocks for online noise and room augmentation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


CONDITIONS = ("clean", "noise", "reverb", "noise_reverb")


def _mono_finite(audio, name):
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1 or not len(value):
        raise ValueError(f"{name} must be a nonempty mono waveform")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def active_speech_mask(audio, frame_length=320, hop_length=160, threshold_db=-40.0):
    """Return samples belonging to frames close to the utterance's peak frame energy."""
    audio = _mono_finite(audio, "speech")
    if frame_length <= 0 or hop_length <= 0 or threshold_db > 0:
        raise ValueError("Invalid active-speech parameters")
    starts = list(range(0, len(audio), hop_length))
    powers = np.array([
        np.mean(audio[start:min(start + frame_length, len(audio))].astype(np.float64) ** 2)
        for start in starts
    ])
    peak = float(powers.max(initial=0.0))
    if peak <= np.finfo(np.float32).tiny:
        raise ValueError("Speech is silent")
    active_frames = powers >= peak * 10.0 ** (threshold_db / 10.0)
    mask = np.zeros(len(audio), dtype=bool)
    for start, active in zip(starts, active_frames):
        if active:
            mask[start:min(start + frame_length, len(audio))] = True
    if not mask.any():
        raise ValueError("No active speech found")
    return mask


def active_speech_rms(audio, **kwargs):
    audio = _mono_finite(audio, "speech")
    mask = active_speech_mask(audio, **kwargs)
    return float(np.sqrt(np.mean(audio[mask].astype(np.float64) ** 2)))


def prepare_rir(rir, leading_threshold_db=-40.0):
    """Trim leading silence and normalize RIR energy while retaining its decay."""
    rir = _mono_finite(rir, "RIR")
    if leading_threshold_db > 0:
        raise ValueError("leading_threshold_db must be non-positive")
    peak = float(np.max(np.abs(rir)))
    if peak <= np.finfo(np.float32).tiny:
        raise ValueError("RIR is silent")
    threshold = peak * 10.0 ** (leading_threshold_db / 20.0)
    start = int(np.flatnonzero(np.abs(rir) >= threshold)[0])
    trimmed = rir[start:]
    energy = float(np.linalg.norm(trimmed.astype(np.float64)))
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("RIR has invalid energy")
    return (trimmed / energy).astype(np.float32), {
        "rir_trimmed_samples": start,
        "rir_energy_scale": 1.0 / energy,
    }


def apply_rir(speech, rir, tail_policy="crop", peak_ceiling=0.99):
    """FFT-convolve speech and RIR, using an explicit output-length policy."""
    speech = _mono_finite(speech, "speech")
    prepared, metadata = prepare_rir(rir)
    reverberant = fftconvolve(speech, prepared, mode="full").astype(np.float32)
    if tail_policy == "crop":
        reverberant = reverberant[:len(speech)]
    elif tail_policy != "full":
        raise ValueError("tail_policy must be 'crop' or 'full'")
    reverberant, limiter_scale = limit_peak(reverberant, peak_ceiling)
    metadata.update({
        "tail_policy": tail_policy,
        "rir_output_samples": len(reverberant),
        "rir_limiter_scale": limiter_scale,
    })
    return reverberant, metadata


def loop_noise(noise, length, crop_start=0):
    """Select a deterministic segment, wrapping short noise as needed."""
    noise = _mono_finite(noise, "noise")
    if length <= 0 or crop_start < 0:
        raise ValueError("Invalid noise segment request")
    indices = (np.arange(length, dtype=np.int64) + int(crop_start)) % len(noise)
    return noise[indices]


def limit_peak(audio, peak_ceiling=0.99):
    audio = _mono_finite(audio, "audio")
    if not 0 < peak_ceiling <= 1:
        raise ValueError("peak_ceiling must be in (0, 1]")
    peak = float(np.max(np.abs(audio)))
    scale = min(1.0, peak_ceiling / peak) if peak else 1.0
    return (audio * scale).astype(np.float32), scale


def mix_at_snr(
    speech,
    noise,
    snr_db,
    crop_start=0,
    global_gain_db=0.0,
    peak_ceiling=0.99,
    active_threshold_db=-40.0,
):
    """Mix noise at an SNR measured only over active speech samples."""
    speech = _mono_finite(speech, "speech")
    if not math.isfinite(snr_db) or not math.isfinite(global_gain_db):
        raise ValueError("SNR and gain must be finite")
    segment = loop_noise(noise, len(speech), crop_start)
    active = active_speech_mask(speech, threshold_db=active_threshold_db)
    speech_power = float(np.mean(speech[active].astype(np.float64) ** 2))
    noise_power = float(np.mean(segment[active].astype(np.float64) ** 2))
    if noise_power <= np.finfo(np.float32).tiny:
        raise ValueError("Selected noise segment is silent")
    target_ratio = 10.0 ** (float(snr_db) / 10.0)
    noise_scale = math.sqrt(speech_power / (noise_power * target_ratio))
    scaled_noise = segment.astype(np.float64) * noise_scale
    gain = 10.0 ** (float(global_gain_db) / 20.0)
    mixed = (speech.astype(np.float64) + scaled_noise) * gain
    mixed, limiter_scale = limit_peak(mixed, peak_ceiling)
    actual_snr = 10.0 * math.log10(
        np.mean(speech[active].astype(np.float64) ** 2)
        / np.mean(scaled_noise[active] ** 2)
    )
    if not np.any(mixed) or not np.isfinite(mixed).all():
        raise ValueError("Augmentation produced invalid or silent audio")
    return mixed, {
        "target_snr_db": float(snr_db),
        "actual_snr_db": actual_snr,
        "noise_crop_start": int(crop_start),
        "noise_wrapped": len(noise) < len(speech) or crop_start + len(speech) > len(noise),
        "noise_scale": noise_scale,
        "global_gain_db": float(global_gain_db),
        "limiter_scale": limiter_scale,
        "peak": float(np.max(np.abs(mixed))),
    }


def apply_global_gain(audio, gain_db=0.0, peak_ceiling=0.99):
    audio = _mono_finite(audio, "audio")
    if not math.isfinite(gain_db):
        raise ValueError("gain_db must be finite")
    gained = audio.astype(np.float64) * 10.0 ** (float(gain_db) / 20.0)
    gained, limiter_scale = limit_peak(gained, peak_ceiling)
    if not np.any(gained):
        raise ValueError("Augmentation produced silence")
    return gained, {"global_gain_db": float(gain_db), "limiter_scale": limiter_scale,
                    "peak": float(np.max(np.abs(gained)))}


def augment_waveform(
    speech,
    condition,
    *,
    noise=None,
    rir=None,
    snr_db=None,
    noise_crop_start=0,
    global_gain_db=0.0,
    tail_policy="crop",
    peak_ceiling=0.99,
):
    """Apply one of the four configured acoustic conditions."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    output = _mono_finite(speech, "speech")
    metadata = {"condition": condition, "input_samples": len(output)}
    if condition in {"reverb", "noise_reverb"}:
        if rir is None:
            raise ValueError("RIR is required for reverberant conditions")
        output, rir_metadata = apply_rir(output, rir, tail_policy, peak_ceiling)
        metadata.update(rir_metadata)
    if condition in {"noise", "noise_reverb"}:
        if noise is None or snr_db is None:
            raise ValueError("Noise and SNR are required for noisy conditions")
        output, noise_metadata = mix_at_snr(
            output, noise, snr_db, noise_crop_start, global_gain_db, peak_ceiling
        )
        metadata.update(noise_metadata)
    else:
        output, gain_metadata = apply_global_gain(output, global_gain_db, peak_ceiling)
        metadata.update(gain_metadata)
    metadata["output_samples"] = len(output)
    return output, metadata


@dataclass(frozen=True)
class AugmentationChoice:
    condition: str
    snr_db: float | None
    global_gain_db: float
    noise_crop_start: int | None
    seed: int


class OnlineAugmentationSampler:
    """Resample augmentation choices deterministically for every sample and epoch."""

    def __init__(self, probabilities, snr_db, gain_db=(-6.0, 3.0), seed=42):
        values = np.array([probabilities.get(name, 0.0) for name in CONDITIONS], dtype=float)
        if np.any(values < 0) or not np.isclose(values.sum(), 1.0):
            raise ValueError("Condition probabilities must be nonnegative and sum to one")
        if not snr_db:
            raise ValueError("snr_db must not be empty")
        if len(gain_db) != 2 or gain_db[0] > gain_db[1]:
            raise ValueError("gain_db must be an ordered [min, max] pair")
        self.probabilities = values
        self.snr_db = tuple(float(value) for value in snr_db)
        self.gain_db = tuple(float(value) for value in gain_db)
        self.seed = int(seed)

    def sample(self, sample_id, epoch, noise_frames=None):
        digest = hashlib.sha256(f"{self.seed}:{epoch}:{sample_id}".encode()).digest()
        derived_seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(derived_seed)
        condition = str(rng.choice(CONDITIONS, p=self.probabilities))
        noisy = condition in {"noise", "noise_reverb"}
        if noisy and noise_frames is not None and noise_frames <= 0:
            raise ValueError("noise_frames must be positive")
        crop_start = None
        if noisy:
            crop_start = int(rng.integers(0, noise_frames)) if noise_frames else 0
        return AugmentationChoice(
            condition=condition,
            snr_db=float(rng.choice(self.snr_db)) if noisy else None,
            global_gain_db=float(rng.uniform(*self.gain_db)),
            noise_crop_start=crop_start,
            seed=derived_seed,
        )
