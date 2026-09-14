#!/usr/bin/env python3
"""Measure realized SNR, length and peak constraints on fixed corruption recipes."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict

import numpy as np

from earnings_asr.audio import load_audio
from earnings_asr.augmentation import augment_waveform
from earnings_asr.common import read_config, read_jsonl, write_json
from earnings_asr.online_data import load_rir_channel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/robust.json")
    parser.add_argument("--manifest", default="data/manifests/robust-v1/corruptions/dev.jsonl")
    parser.add_argument("--per-cell", type=int, default=10)
    parser.add_argument("--output", default="reports/augmentation-audit.json")
    return parser.parse_args()


def stable_rows(rows, seed):
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest())


def render(row, settings):
    speech = load_audio(row["speech_audio"])
    noise = load_audio(row["noise_audio"])
    rir = (load_rir_channel(row["rir_audio"], row["rir_channel"])
           if row["rir_audio"] else None)
    return augment_waveform(
        speech,
        row["condition"],
        noise=noise,
        rir=rir,
        snr_db=row["snr_db"],
        noise_crop_start=row["noise_crop_start"],
        global_gain_db=row["global_gain_db"],
        tail_policy=settings["rir_tail_policy"],
        peak_ceiling=settings["peak_ceiling"],
    )


def main():
    args = parse_args()
    if args.per_cell <= 0:
        raise ValueError("per-cell must be positive")
    config = read_config(args.config)
    candidates = [row for row in read_jsonl(args.manifest)
                  if row["condition"] in {"noise", "noise_reverb"}]
    cells = defaultdict(list)
    for row in candidates:
        cells[(row["condition"], row["snr_db"])].append(row)
    selected = []
    for key in sorted(cells):
        selected.extend(stable_rows(cells[key], config["seed"])[:args.per_cell])

    errors, peaks, details = [], [], []
    limiter_count = wrapped_count = 0
    for row in selected:
        audio, measured = render(row, config["robust_data"])
        error = measured["actual_snr_db"] - row["snr_db"]
        errors.append(error)
        peaks.append(float(np.max(np.abs(audio))))
        limiter_count += measured["limiter_scale"] < 1.0
        wrapped_count += measured["noise_wrapped"]
        details.append({
            "id": row["id"],
            "condition": row["condition"],
            "snr_db": row["snr_db"],
            "actual_snr_db": measured["actual_snr_db"],
            "snr_error_db": error,
            "peak": measured["peak"],
            "limiter_scale": measured["limiter_scale"],
            "noise_wrapped": measured["noise_wrapped"],
            "samples": len(audio),
        })
    absolute = np.abs(errors)
    report = {
        "manifest": args.manifest,
        "examples": len(details),
        "per_cell": args.per_cell,
        "cells": {f"{condition}:{snr}dB": count for (condition, snr), count
                  in sorted(Counter((row["condition"], row["snr_db"]) for row in selected).items())},
        "snr_error_db": {
            "mean_absolute": float(np.mean(absolute)),
            "p95_absolute": float(np.percentile(absolute, 95)),
            "maximum_absolute": float(np.max(absolute)),
        },
        "maximum_peak": max(peaks),
        "limiter_examples": int(limiter_count),
        "wrapped_noise_examples": int(wrapped_count),
        "all_finite": all(np.isfinite(item["peak"]) for item in details),
        "details": details,
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()
