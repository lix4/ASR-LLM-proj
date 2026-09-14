import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from earnings_asr.common import read_jsonl, write_jsonl
from earnings_asr.data import (
    audit_disjoint,
    build_spgi,
    exclude_cross_split_overlaps,
    export_sft,
    select_by_duration,
)


def test_sampling_reproducible_order_independent_and_duration_based():
    rows = [{"id": str(i), "duration": i + 1} for i in range(20)]
    selected = select_by_duration(rows, 0.02, 42)
    assert selected == select_by_duration(list(reversed(rows)), 0.02, 42)
    assert sum(r["duration"] for r in selected) >= 72
    assert sum(r["duration"] for r in selected[:-1]) < 72
    with pytest.raises(ValueError, match="Insufficient"):
        select_by_duration(rows, 1, 42)


@pytest.mark.parametrize("key", ["id", "audio_sha256", "recording_id"])
def test_cross_split_leakage_detected(key):
    with pytest.raises(ValueError, match="Cross-split"):
        audit_disjoint({"train": [{key: "same"}], "test": [{key: "same"}]})


def test_cross_split_conflicts_are_removed_from_every_side():
    splits = {
        "train": [{"id": "a", "audio_sha256": "same", "recording_id": "r1"}],
        "dev": [{"id": "b", "audio_sha256": "same", "recording_id": "r2"}],
        "test": [{"id": "c", "audio_sha256": "unique", "recording_id": "r3"}],
    }
    cleaned, report = exclude_cross_split_overlaps(splits)
    assert cleaned == {"train": [], "dev": [], "test": splits["test"]}
    assert report["groups"] == {"id": 0, "audio_sha256": 1, "recording_id": 0}
    assert report["excluded_instances"]["train"] == 1


def test_parquet_to_manifests_and_sft(tmp_path):
    config = {"data_dir": str(tmp_path), "dataset_revision": "fixture", "seed": 42,
              "min_duration": 0.5, "max_duration": 30, "hours": {"train": 1 / 3600, "dev": 1 / 3600, "test": 1 / 3600}}
    for i, (folder, split) in enumerate((("S", "train"), ("dev", "validation"), ("test", "test"))):
        wave = np.random.default_rng(i).normal(0, 0.1, 16000).astype("float32")
        stream = io.BytesIO()
        sf.write(stream, wave, 16000, format="WAV")
        valid = {"wav_filename": f"recording{i}-0001.wav", "transcript": "fixture transcript",
                 "audio": {"bytes": stream.getvalue(), "path": "test.wav"}}
        invalid = {**valid, "wav_filename": "empty.wav", "transcript": ""}
        duplicate = {**valid, "wav_filename": f"duplicate-{i}.wav"}
        path = tmp_path / "spgispeech" / "fixture" / folder / f"{split}-00000.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([valid, invalid, duplicate]), path)
    dest = tmp_path / "manifests" / "test-run"
    report = build_spgi(config, dest, r"(recording\d+)-")
    assert report["splits"]["train"]["filtered"] == {"empty_text": 1, "duplicate_id_or_audio": 1}
    assert report["splits"]["train"]["samples"] == 1
    rows = read_jsonl(dest / "train.jsonl")
    assert Path(rows[0]["audio"]).is_file()
    assert sf.info(rows[0]["audio"]).duration == 1
    export_sft(dest / "train.jsonl", tmp_path / "sft.jsonl")
    assert read_jsonl(tmp_path / "sft.jsonl")[0]["text"].startswith("language English<asr_text>")
    with pytest.raises(ValueError, match="train-only"):
        export_sft(dest / "test.jsonl", tmp_path / "bad.jsonl")
    with pytest.raises(FileExistsError):
        build_spgi(config, dest)


def test_sft_rejects_missing_audio(tmp_path):
    write_jsonl(tmp_path / "manifest.jsonl", [{"split": "train", "audio": "missing.wav", "text": "hello"}])
    with pytest.raises(ValueError, match="Missing"):
        export_sft(tmp_path / "manifest.jsonl", tmp_path / "sft.jsonl")
