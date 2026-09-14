#!/usr/bin/env python3
"""Render one fixed dev example for listening and waveform/spectrogram review."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw
from scipy.signal import spectrogram

from earnings_asr.audio import SAMPLE_RATE, load_audio
from earnings_asr.augmentation import augment_waveform
from earnings_asr.common import read_config, read_jsonl, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/robust.json")
    parser.add_argument("--manifest", default="data/manifests/robust-v1/corruptions/dev.jsonl")
    parser.add_argument("--output", default="outputs/augmentation-audit")
    parser.add_argument("--speech-id")
    parser.add_argument("--snr", type=float, default=0.0)
    return parser.parse_args()


def select_rows(rows, speech_id, snr):
    if speech_id is None:
        speech_id = rows[0]["speech_id"]
    chosen = []
    for condition in ("clean", "noise", "reverb", "noise_reverb"):
        matches = [row for row in rows if row["speech_id"] == speech_id
                   and row["condition"] == condition
                   and (row["snr_db"] is None or row["snr_db"] == snr)]
        if len(matches) != 1:
            raise ValueError(f"Expected one {condition} row for {speech_id} at {snr} dB")
        chosen.append(matches[0])
    return chosen


def load_rir_channel(path, channel):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or channel is None or not 0 <= channel < audio.shape[1]:
        raise ValueError(f"Invalid RIR channel request: {path} channel {channel}")
    return audio[:, channel]


def render_row(row, settings):
    speech = load_audio(row["speech_audio"])
    noise = load_audio(row["noise_audio"]) if row["noise_audio"] else None
    rir = (load_rir_channel(row["rir_audio"], row["rir_channel"])
           if row["rir_audio"] else None)
    return augment_waveform(
        speech,
        row["condition"],
        noise=noise,
        rir=rir,
        snr_db=row["snr_db"],
        noise_crop_start=row["noise_crop_start"] or 0,
        global_gain_db=row["global_gain_db"],
        tail_policy=settings["rir_tail_policy"],
        peak_ceiling=settings["peak_ceiling"],
    )


def colorize(values):
    """Small dependency-free blue/yellow heat map for normalized [0, 1] values."""
    values = np.clip(values, 0, 1)
    red = np.clip(3.0 * values - 1.0, 0, 1)
    green = np.clip(3.0 * values, 0, 1)
    blue = np.clip(1.5 - 2.0 * values, 0, 1)
    return np.stack((red, green, blue), axis=-1).astype(np.float32)


def make_audit_plot(items, output):
    width, label_width, wave_height, spec_height = 1400, 170, 100, 180
    row_height = wave_height + spec_height + 30
    canvas = Image.new("RGB", (width, row_height * len(items)), "white")
    draw = ImageDraw.Draw(canvas)
    plot_width = width - label_width - 20
    for index, (label, audio) in enumerate(items):
        top = index * row_height
        draw.text((12, top + 10), label, fill="black")
        draw.text((12, top + 30), f"{len(audio) / SAMPLE_RATE:.2f} s", fill="black")
        points = []
        edges = np.linspace(0, len(audio), plot_width + 1).astype(int)
        for x, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            peak = float(np.max(np.abs(audio[left:max(left + 1, right)])))
            points.append((label_width + x, top + wave_height // 2 - peak * (wave_height / 2 - 5)))
        draw.line(points, fill=(15, 80, 150), width=1)
        _, _, magnitude = spectrogram(audio, fs=SAMPLE_RATE, nperseg=400, noverlap=240,
                                      mode="magnitude")
        db = 20 * np.log10(np.maximum(magnitude, 1e-8))
        db -= db.max()
        normalized = np.clip((db + 80.0) / 80.0, 0, 1)
        pixels = (colorize(normalized[::-1]) * 255).astype("uint8")
        image = Image.fromarray(pixels, mode="RGB").resize((plot_width, spec_height))
        canvas.paste(image, (label_width, top + wave_height))
    canvas.save(output)


def main():
    args = parse_args()
    config = read_config(args.config)
    rows = select_rows(read_jsonl(args.manifest), args.speech_id, args.snr)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    output.mkdir(parents=True)
    rendered, metadata = [], {"manifest": str(Path(args.manifest).resolve()), "examples": []}
    for index, row in enumerate(rows):
        audio, parameters = render_row(row, config["robust_data"])
        filename = f"{index:02d}_{row['condition']}.wav"
        sf.write(output / filename, audio, SAMPLE_RATE, subtype="PCM_16")
        rendered.append((row["condition"], audio))
        metadata["examples"].append({"file": filename, "corruption": row, "measured": parameters})
    silence = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    comparison = np.concatenate([part for _, audio in rendered for part in (audio, silence)])
    sf.write(output / "comparison.wav", comparison, SAMPLE_RATE, subtype="PCM_16")
    make_audit_plot(rendered, output / "waveform_spectrogram.png")
    metadata["transcript"] = rows[0]["text"]
    write_json(output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
