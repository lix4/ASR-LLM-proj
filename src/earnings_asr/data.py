"""Explicit SPGISpeech shard selection, validation, deterministic sampling and SFT export."""

import hashlib
import io
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url

from .common import read_jsonl, sha256_file, write_json, write_jsonl

SPLIT_PATTERNS = {"train": "S/train-", "dev": "dev/validation-", "test": "test/test-"}


def shard_plan(config):
    info = HfApi().dataset_info(config["dataset_id"], revision=config["dataset_revision"],
                              files_metadata=True, timeout=30)
    result = {}
    for split, prefix in SPLIT_PATTERNS.items():
        result[split] = [{"filename": f.rfilename, "bytes": f.size} for f in info.siblings
                         if f.rfilename.startswith(prefix) and f.rfilename.endswith(".parquet")]
        result[split].sort(key=lambda item: item["filename"])
        if not result[split]:
            raise ValueError(f"No shards found for {split}; inspect the upstream layout")
    return result


def download_spgi(config):
    plan = shard_plan(config)
    first = plan["train"][0]["filename"]
    try:
        get_hf_file_metadata(hf_hub_url(config["dataset_id"], first, repo_type="dataset",
                                      revision=config["dataset_revision"]), token=get_token(), timeout=30)
    except Exception as error:
        raise RuntimeError("SPGISpeech access unavailable. Accept the dataset terms on Hugging Face "
                           "and run `hf auth login` in your terminal; never put tokens in config.") from error
    root = Path(config["data_dir"]) / "spgispeech" / config["dataset_revision"]
    root.mkdir(parents=True, exist_ok=True)
    remaining = sum(item["bytes"] or 0 for items in plan.values() for item in items
                    if not (root / item["filename"]).exists())
    if shutil.disk_usage(root).free < remaining + 15 * 1024**3:
        raise RuntimeError("Insufficient disk space for selected shards plus extracted experiment audio")
    for items in plan.values():
        for item in items:
            hf_hub_download(config["dataset_id"], item["filename"], repo_type="dataset",
                            revision=config["dataset_revision"], local_dir=root)
    write_json(root / "download-plan.json", plan)
    return root


def select_by_duration(rows, target_hours, seed):
    if not math.isfinite(target_hours) or target_hours <= 0:
        raise ValueError("Target hours must be positive and finite")
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}:{row['id']}".encode()).digest())
    selected, seconds = [], 0.0
    for row in ordered:
        selected.append(row)
        seconds += row["duration"]
        if seconds >= target_hours * 3600:
            return selected
    raise ValueError(f"Insufficient valid audio: {seconds / 3600:.4f} h < {target_hours} h")


def audit_disjoint(splits):
    owners = {key: {} for key in ("id", "audio_sha256", "recording_id")}
    for split, rows in splits.items():
        for row in rows:
            for key, lookup in owners.items():
                value = row.get(key)
                if value is None:
                    continue
                previous = lookup.setdefault(value, split)
                if previous != split:
                    raise ValueError(f"Cross-split overlap detected for {key}: {previous}/{split}")


def exclude_cross_split_overlaps(splits):
    """Remove every occurrence of an ID/audio/recording shared by official splits."""
    memberships = {key: {} for key in ("id", "audio_sha256", "recording_id")}
    for split, rows in splits.items():
        for row in rows:
            for key, lookup in memberships.items():
                value = row.get(key)
                if value is not None:
                    lookup.setdefault(value, set()).add(split)
    conflicts = {key: {value for value, owners in lookup.items() if len(owners) > 1}
                 for key, lookup in memberships.items()}
    excluded = Counter()
    cleaned = {}
    for split, rows in splits.items():
        cleaned[split] = []
        for row in rows:
            reasons = [key for key, values in conflicts.items() if row.get(key) in values]
            if reasons:
                excluded[split] += 1
                for reason in reasons:
                    excluded[f"{split}:{reason}"] += 1
            else:
                cleaned[split].append(row)
    return cleaned, {"groups": {key: len(values) for key, values in conflicts.items()},
                     "excluded_instances": dict(excluded)}


def audio_bytes(row):
    audio = row.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        raise ValueError("Expected Parquet audio struct with embedded bytes; inspect dataset schema")
    return audio["bytes"]


def scan_shards(paths, split, config, recording_pattern=None):
    rows, rejected, seen_ids, seen_audio = [], Counter(), set(), set()
    pattern = re.compile(recording_pattern) if recording_pattern else None
    for path in paths:
        parquet = pq.ParquetFile(path)
        required = {"wav_filename", "audio", "transcript"}
        if not required.issubset(parquet.schema_arrow.names):
            raise ValueError(f"Unexpected schema in {path.name}: {parquet.schema_arrow.names}")
        index = 0
        for batch in parquet.iter_batches(batch_size=32, columns=sorted(required)):
            for row in batch.to_pylist():
                source_index = index
                index += 1
                sample_id, text = row["wav_filename"], row["transcript"]
                if not isinstance(sample_id, str) or not sample_id.strip():
                    rejected["empty_id"] += 1
                    continue
                if not isinstance(text, str) or not text.strip():
                    rejected["empty_text"] += 1
                    continue
                try:
                    waveform, rate = sf.read(io.BytesIO(audio_bytes(row)), dtype="float32", always_2d=True)
                except (ValueError, RuntimeError):
                    rejected["corrupt_audio"] += 1
                    continue
                if rate != 16000 or waveform.shape[1] != 1:
                    rejected["unexpected_audio_format"] += 1
                    continue
                duration = len(waveform) / rate
                if not config["min_duration"] <= duration <= config["max_duration"]:
                    rejected["duration_out_of_range"] += 1
                    continue
                if not np.isfinite(waveform).all():
                    rejected["non_finite_audio"] += 1
                    continue
                if not np.any(waveform):
                    rejected["digital_silence_with_text"] += 1
                    continue
                digest = hashlib.sha256(waveform.astype("<f4").tobytes()).hexdigest()
                if sample_id in seen_ids or digest in seen_audio:
                    rejected["duplicate_id_or_audio"] += 1
                    continue
                recording_id = None
                if pattern:
                    match = pattern.search(sample_id)
                    if not match:
                        raise ValueError("Recording-ID pattern does not match every filename")
                    recording_id = match.group(1)
                seen_ids.add(sample_id)
                seen_audio.add(digest)
                rows.append({"id": sample_id, "text": text, "duration": duration, "split": split,
                             "recording_id": recording_id, "audio_sha256": digest,
                             "source_shard": str(path), "source_row": source_index})
    return rows, dict(rejected)


def build_spgi(config, destination=None, recording_pattern=None):
    root = Path(config["data_dir"]) / "spgispeech" / config["dataset_revision"]
    destination = Path(destination or Path(config["data_dir"]) / "manifests" / "spgi-v1")
    if destination.exists():
        raise FileExistsError(f"{destination} exists. Use a new destination to preserve fixed manifests.")
    recording_pattern = recording_pattern or config.get("recording_id_pattern")
    splits, report = {}, {"seed": config["seed"], "dataset_revision": config["dataset_revision"],
                           "recording_id_pattern": recording_pattern, "splits": {}}
    for split, prefix in SPLIT_PATTERNS.items():
        paths = sorted(root.glob(prefix + "*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No {split} shards found. Run download-data first.")
        rows, rejected = scan_shards(paths, split, config, recording_pattern)
        splits[split] = rows
        report["splits"][split] = {"scanned_valid": len(rows), "filtered": rejected}
    # Remove every side of upstream cross-split conflicts, then prove the pool is disjoint.
    splits, cross_split = exclude_cross_split_overlaps(splits)
    audit_disjoint(splits)
    selected = {split: select_by_duration(rows, config["hours"][split], config["seed"])
                for split, rows in splits.items()}
    audio_root = Path(config["data_dir"]) / "audio" / destination.name
    for split, rows in selected.items():
        lookup = {(row["source_shard"], row["source_row"]): row for row in rows}
        for shard in sorted({row["source_shard"] for row in rows}):
            index = 0
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=32, columns=["audio"]):
                for item in batch.to_pylist():
                    row = lookup.get((shard, index))
                    index += 1
                    if row is None:
                        continue
                    output = audio_root / split / (hashlib.sha256(row["id"].encode()).hexdigest() + ".wav")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    # Preserve the source WAV bytes; no lossy re-encoding.
                    output.write_bytes(audio_bytes(item))
                    row["audio"] = str(output.resolve())
        manifest = destination / f"{split}.jsonl"
        write_jsonl(manifest, rows)
        durations = [row["duration"] for row in rows]
        report["splits"][split].update({"samples": len(rows), "hours": sum(durations) / 3600,
            "duration_percentiles": dict(zip(("min", "p50", "p95", "max"),
                                              np.percentile(durations, [0, 50, 95, 100]).tolist())),
            "manifest_sha256": sha256_file(manifest),
            "missing_recording_id": sum(row["recording_id"] is None for row in rows)})
    report["cross_split_conflicts"] = cross_split
    report["duplicate_audit"] = "All occurrences of cross-split ID/audio/recording conflicts excluded before sampling"
    report["limitations"] = ["Recording ID is the inspected first path component of wav_filename.",
                            "Semantic audio/transcript matching requires manual review.",
                            "No claim that the base model has never seen these public data."]
    write_json(destination / "report.json", report)
    return report


def export_sft(manifest, output):
    rows = read_jsonl(manifest)
    splits = {row["split"] for row in rows}
    if not rows or not splits.issubset({"train", "dev"}) or len(splits) != 1:
        raise ValueError("SFT export requires a nonempty train-only or dev-only manifest")
    result = []
    for row in rows:
        path = Path(row["audio"])
        if not path.is_file() or not row["text"].strip():
            raise ValueError("Missing local audio or empty transcript")
        result.append({"id": row["id"], "split": row["split"],
                       "audio": str(path.resolve()),
                       "text": "language English<asr_text>" + row["text"].strip(),
                       "frames": row.get("frames"), "duration": row.get("duration")})
    write_jsonl(output, result)
    return {"samples": len(result), "split": next(iter(splits)), "manifest_sha256": sha256_file(manifest)}
