"""Create deterministic synthetic-room RIR listening examples from one manifest item."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass(frozen=True)
class RoomProfile:
    name: str
    rt60_seconds: float
    drr_db: float
    early_delays_ms: tuple[float, ...]
    seed: int


PROFILES = (
    RoomProfile("small_office", 0.25, 6.0, (7, 11, 17, 23, 31, 43), 101),
    RoomProfile("conference_room", 0.55, 1.0, (12, 19, 28, 37, 52, 71, 95), 202),
    RoomProfile("large_room", 1.20, -3.0, (20, 32, 47, 66, 89, 117, 151), 303),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/spgi-v1/dev.jsonl")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", default="outputs/rir-demo")
    return parser.parse_args()


def find_sample(manifest: Path, sample_id: str) -> dict:
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] == sample_id:
                return row
    raise ValueError(f"Sample ID not found: {sample_id}")


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def normalize_to_rms(audio: np.ndarray, target_rms: float, peak_limit: float = 0.95):
    level = rms(audio)
    if not math.isfinite(level) or level == 0:
        raise ValueError("Audio is silent or invalid")
    gain = target_rms / level
    peak = float(np.max(np.abs(audio)))
    if peak * gain > peak_limit:
        gain = peak_limit / peak
    return (audio * gain).astype(np.float32), float(gain)


def synthesize_rir(profile: RoomProfile, sample_rate: int) -> np.ndarray:
    """Build direct sound, discrete early reflections and diffuse exponential decay."""
    length = round((profile.rt60_seconds + 0.08) * sample_rate)
    reverb = np.zeros(length, dtype=np.float64)
    rng = np.random.default_rng(profile.seed)

    for index, delay_ms in enumerate(profile.early_delays_ms):
        sample = round(delay_ms * sample_rate / 1000)
        decay = math.exp(-6.907755 * (sample / sample_rate) / profile.rt60_seconds)
        polarity = -1.0 if index % 3 == 1 else 1.0
        reverb[sample] += polarity * decay * rng.uniform(0.55, 0.95)

    late_start = round(profile.early_delays_ms[0] * sample_rate / 1000)
    noise = rng.standard_normal(length)
    noise = signal.sosfilt(signal.butter(2, 0.72, output="sos"), noise)
    time = np.arange(length) / sample_rate
    envelope = np.exp(-6.907755 * time / profile.rt60_seconds)
    fade = np.zeros(length)
    fade_length = round(0.025 * sample_rate)
    fade[late_start : late_start + fade_length] = np.linspace(0, 1, fade_length, endpoint=False)
    fade[late_start + fade_length :] = 1
    reverb += 0.12 * noise * envelope * fade

    target_reverb_energy = 10 ** (-profile.drr_db / 10)
    energy = float(np.sum(reverb**2))
    if energy == 0:
        raise RuntimeError("Synthesized RIR has no reverberant energy")
    reverb *= math.sqrt(target_reverb_energy / energy)

    rir = reverb
    rir[0] += 1.0
    return rir.astype(np.float32)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    sample = find_sample(Path(args.manifest), args.sample_id)
    source, sample_rate = sf.read(sample["audio"], dtype="float32", always_2d=True)
    if sample_rate != 16000 or source.shape[1] != 1:
        raise ValueError("Expected 16 kHz mono SPGISpeech audio")
    source = source[:, 0]
    target_rms = 10 ** (-24 / 20)
    dry, dry_gain = normalize_to_rms(source, target_rms)
    sf.write(output / "00_dry.wav", dry, sample_rate, subtype="PCM_16")

    comparison = [dry]
    silence = np.zeros(sample_rate, dtype=np.float32)
    metadata = {
        "source": {
            "manifest": str(Path(args.manifest)),
            "sample_id": sample["id"],
            "duration_seconds": sample["duration"],
            "transcript": sample["text"],
        },
        "sample_rate": sample_rate,
        "channels": 1,
        "dry_target_rms_dbfs": -24.0,
        "dry_gain": dry_gain,
        "profiles": [],
        "comparison_order": ["00_dry.wav"],
        "note": "Synthetic RIRs for local listening; these are not measured room responses.",
    }

    for number, profile in enumerate(PROFILES, start=1):
        rir = synthesize_rir(profile, sample_rate)
        wet = signal.fftconvolve(dry, rir, mode="full").astype(np.float32)
        wet, wet_gain = normalize_to_rms(wet, target_rms)
        rir_name = f"rir_{profile.name}.wav"
        wet_name = f"{number:02d}_{profile.name}.wav"
        sf.write(output / rir_name, rir, sample_rate, subtype="FLOAT")
        sf.write(output / wet_name, wet, sample_rate, subtype="PCM_16")
        comparison.extend((silence, wet))
        metadata["profiles"].append(
            {**asdict(profile), "rir_file": rir_name, "audio_file": wet_name,
             "wet_gain": wet_gain, "output_peak": float(np.max(np.abs(wet))),
             "output_rms_dbfs": 20 * math.log10(rms(wet))}
        )
        metadata["comparison_order"].append(wet_name)

    sf.write(output / "comparison.wav", np.concatenate(comparison), sample_rate, subtype="PCM_16")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
