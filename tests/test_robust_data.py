from pathlib import Path

import pytest

from earnings_asr.common import read_jsonl, write_jsonl
from earnings_asr.robust_data import (
    _audit_group_disjoint,
    _partition,
    _rir_source_and_group,
    build_corruption_manifest,
)


def test_source_groups_are_held_out_and_partition_is_stable():
    assert _rir_source_and_group(Path("simulated_rirs/smallroom/rir.wav")) == (
        "simulated", "simulated:smallroom"
    )
    assert _rir_source_and_group(Path("real_rirs_isotropic_noises/AIR/rir.wav")) == (
        "real", "real:air"
    )
    ratios = {"train": 0.8, "dev": 0.1, "test": 0.1}
    assert _partition(42, "source.wav", ratios) == _partition(42, "source.wav", ratios)
    with pytest.raises(ValueError, match="leakage"):
        _audit_group_disjoint(
            {"train": [{"group": "same"}], "dev": [{"group": "same"}], "test": []},
            "group",
        )


def test_fixed_corruption_manifest_is_complete_and_reproducible(tmp_path):
    config = {
        "seed": 42,
        "robust_data": {"snr_db": [-5, 10]},
    }
    speech = [
        {"id": f"speech-{index}", "audio": f"speech-{index}.flac", "text": "hello",
         "split": "dev"}
        for index in range(2)
    ]
    noise = [
        {"id": f"noise-{index}", "audio": f"noise-{index}.wav", "frames": 1000 + index,
         "split": "dev"}
        for index in range(2)
    ]
    rir = [
        {"id": f"rir-{index}", "audio": f"rir-{index}.wav", "split": "dev"}
        for index in range(2)
    ]
    for name, rows in (("speech", speech), ("noise", noise), ("rir", rir)):
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    output = tmp_path / "corruptions.jsonl"
    first_report = build_corruption_manifest(
        config, "dev", tmp_path / "speech.jsonl", tmp_path / "noise.jsonl",
        tmp_path / "rir.jsonl", output
    )
    first = read_jsonl(output)
    second_report = build_corruption_manifest(
        config, "dev", tmp_path / "speech.jsonl", tmp_path / "noise.jsonl",
        tmp_path / "rir.jsonl", output
    )
    assert first == read_jsonl(output)
    assert first_report == second_report
    assert len(first) == 12
    assert len({row["id"] for row in first}) == len(first)
    assert {row["condition"] for row in first} == {
        "clean", "noise", "reverb", "noise_reverb"
    }
    noisy = [row for row in first if row["noise_id"]]
    assert all(0 <= row["noise_crop_start"] < 1001 for row in noisy)
    assert all(row["seed"] is not None for row in first)
