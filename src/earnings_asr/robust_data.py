"""Manifest and fixed-corruption preparation for LibriSpeech, MUSAN and SLR28."""

from __future__ import annotations

import hashlib
import json
import math
import shlex
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import soundfile as sf

from .common import read_jsonl, sha256_file, write_json, write_jsonl
from .data import select_by_duration


LIBRISPEECH_SPLITS = {
    "train": "train-clean-100",
    "dev": "dev-clean",
    "test_clean": "test-clean",
    "test_other": "test-other",
}


def _settings(config):
    if "robust_data" not in config:
        raise ValueError("Config has no robust_data section; use configs/robust.json")
    return config["robust_data"]


def _resolve(path):
    return Path(path).expanduser().resolve()


def _audio_metadata(path, expected_rate=16000, require_mono=True):
    info = sf.info(path)
    invalid_channels = require_mono and info.channels != 1
    if info.frames <= 0 or invalid_channels or info.samplerate != expected_rate:
        raise ValueError(
            f"Unexpected audio format for {path}: {info.frames} frames, "
            f"{info.channels} channels, {info.samplerate} Hz"
        )
    return {"frames": int(info.frames), "duration": info.frames / info.samplerate,
            "sample_rate": int(info.samplerate), "channels": int(info.channels)}


def _stable_fraction(seed, group):
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _partition(seed, group, ratios):
    if set(ratios) != {"train", "dev", "test"} or not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Split ratios must contain train/dev/test and sum to one")
    point = _stable_fraction(seed, group)
    if point < ratios["train"]:
        return "train"
    if point < ratios["train"] + ratios["dev"]:
        return "dev"
    return "test"


def archive_inventory(config):
    settings = _settings(config)
    raw_dir = _resolve(settings["raw_dir"])
    result = []
    for filename, url in settings["archives"].items():
        path = raw_dir / filename
        result.append({
            "filename": filename,
            "url": url,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        })
    return result


def _safe_target(root, member_name):
    root = root.resolve()
    target = (root / member_name).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Archive member escapes destination: {member_name}")


def extract_archives(config):
    """Extract each verified local archive once and record source hashes."""
    settings = _settings(config)
    raw_dir = _resolve(settings["raw_dir"])
    corpus_dir = _resolve(settings["corpus_dir"])
    corpus_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for filename in settings["archives"]:
        archive = raw_dir / filename
        if not archive.is_file():
            raise FileNotFoundError(f"Missing archive: {archive}")
        digest = sha256_file(archive)
        marker = corpus_dir / f".{filename}.extracted.json"
        if marker.is_file():
            previous = json.loads(marker.read_text())
            if previous.get("sha256") == digest:
                records.append(previous)
                continue
            raise ValueError(f"Archive changed after extraction: {archive}")
        if filename.endswith(".tar.gz"):
            with tarfile.open(archive, "r:gz") as stream:
                for member in stream.getmembers():
                    _safe_target(corpus_dir, member.name)
                stream.extractall(corpus_dir, filter="data")
        elif filename.endswith(".zip"):
            with zipfile.ZipFile(archive) as stream:
                for member in stream.infolist():
                    _safe_target(corpus_dir, member.filename)
                stream.extractall(corpus_dir)
        else:
            raise ValueError(f"Unsupported archive type: {filename}")
        record = {"archive": str(archive), "sha256": digest}
        write_json(marker, record)
        records.append(record)
    write_json(corpus_dir / "extraction-report.json", records)
    return records


def _load_librispeech_transcripts(split_root):
    transcripts = {}
    for path in sorted(split_root.rglob("*.trans.txt")):
        for line in path.read_text().splitlines():
            sample_id, separator, text = line.partition(" ")
            if not separator or not text.strip() or sample_id in transcripts:
                raise ValueError(f"Malformed or duplicate transcript in {path}")
            transcripts[sample_id] = text.strip()
    return transcripts


def build_speech_manifests(config, destination):
    settings = _settings(config)
    root = _resolve(settings["corpus_dir"]) / "LibriSpeech"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = {"source": "OpenSLR SLR12", "splits": {}, "train_hours_target": settings["train_hours"]}
    seen_ids = set()
    for split, folder in LIBRISPEECH_SPLITS.items():
        split_root = root / folder
        transcripts = _load_librispeech_transcripts(split_root)
        rows = []
        for path in sorted(split_root.rglob("*.flac")):
            sample_id = path.stem
            parts = sample_id.split("-")
            if len(parts) != 3 or sample_id not in transcripts:
                raise ValueError(f"Missing transcript or invalid LibriSpeech ID: {path}")
            if sample_id in seen_ids:
                raise ValueError(f"Cross-split speech ID overlap: {sample_id}")
            seen_ids.add(sample_id)
            metadata = _audio_metadata(path, settings["sample_rate"])
            rows.append({
                "id": sample_id,
                "audio": str(path.resolve()),
                "text": transcripts[sample_id],
                "speaker_id": parts[0],
                "chapter_id": "-".join(parts[:2]),
                "official_subset": folder,
                "split": split,
                **metadata,
            })
        if len(rows) != len(transcripts):
            raise ValueError(f"Audio/transcript count mismatch in {split_root}")
        if split == "train":
            rows = select_by_duration(rows, settings["train_hours"], config["seed"])
        manifest = destination / f"{split}.jsonl"
        write_jsonl(manifest, rows)
        report["splits"][split] = {
            "files": len(rows),
            "hours": sum(row["duration"] for row in rows) / 3600.0,
            "speakers": len({row["speaker_id"] for row in rows}),
            "manifest_sha256": sha256_file(manifest),
        }
    write_json(destination / "report.json", report)
    return report


def build_noise_manifests(config, destination):
    settings = _settings(config)
    root = _resolve(settings["corpus_dir"]) / "musan"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows_by_split = {split: [] for split in ("train", "dev", "test")}
    for path in sorted(root.rglob("*.wav")):
        relative = path.relative_to(root)
        category = relative.parts[0]
        if category not in {"noise", "music", "speech"}:
            raise ValueError(f"Unexpected MUSAN category: {path}")
        group_id = relative.as_posix()
        split = _partition(config["seed"], group_id, settings["noise_split_ratios"])
        rows_by_split[split].append({
            "id": "musan:" + relative.with_suffix("").as_posix(),
            "audio": str(path.resolve()),
            "source_recording_id": group_id,
            "category": category,
            "split": split,
            **_audio_metadata(path, settings["sample_rate"]),
        })
    if not all(rows_by_split.values()):
        raise ValueError("MUSAN split is empty; inspect the extracted corpus")
    _audit_group_disjoint(rows_by_split, "source_recording_id")
    report = {"source": "OpenSLR SLR17", "split_method": "SHA-256 by complete source file",
              "splits": {}}
    for split, rows in rows_by_split.items():
        manifest = destination / f"{split}.jsonl"
        write_jsonl(manifest, rows)
        report["splits"][split] = {
            "files": len(rows),
            "hours": sum(row["duration"] for row in rows) / 3600.0,
            "categories": dict(Counter(row["category"] for row in rows)),
            "manifest_sha256": sha256_file(manifest),
        }
    write_json(destination / "report.json", report)
    return report


def _rir_source_and_group(relative):
    lower = relative.as_posix().lower()
    if "simulated_rirs" in lower:
        size = next((name for name in ("smallroom", "mediumroom", "largeroom") if name in lower),
                    "unknownroom")
        # SLR28 simulated files encode room parameters in each filename. Group by room class so
        # entire room-size families are held out together.
        return "simulated", f"simulated:{size}"
    if "air" in lower:
        return "real", "real:air"
    if "rwcp" in lower:
        return "real", "real:rwcp"
    if "rvb" in lower or "reverb" in lower:
        return "real", "real:reverb2014"
    return "real", "real:other"


def _rir_split(source_group):
    explicit = {
        "simulated:smallroom": "train",
        "simulated:mediumroom": "dev",
        "simulated:largeroom": "test",
        "real:reverb2014": "train",
        "real:rwcp": "dev",
        "real:air": "test",
    }
    return explicit.get(source_group, "train")


def _listed_rirs(root):
    listed = {}
    for list_path in sorted(root.rglob("rir_list")):
        for line in list_path.read_text().splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            tokens = shlex.split(value)
            path_value = tokens[-1]
            room_id = tokens[tokens.index("--room-id") + 1] if "--room-id" in tokens else None
            rir_id = tokens[tokens.index("--rir-id") + 1] if "--rir-id" in tokens else None
            candidates = (
                (root / path_value).resolve(),
                (root.parent / path_value).resolve(),
                (list_path.parent / path_value).resolve(),
            )
            candidate = next((item for item in candidates if item.is_file()), None)
            if candidate is None:
                raise FileNotFoundError(f"RIR listed by {list_path} is missing: {path_value}")
            if candidate in listed:
                raise ValueError(f"RIR appears in multiple list entries: {candidate}")
            listed[candidate] = {"path": candidate, "room_id": room_id, "rir_id": rir_id}
    if not listed:
        raise ValueError(f"No rir_list files found under {root}")
    return [listed[path] for path in sorted(listed)]


def build_rir_manifests(config, destination):
    settings = _settings(config)
    root = _resolve(settings["corpus_dir"]) / "RIRS_NOISES"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows_by_split = {split: [] for split in ("train", "dev", "test")}
    for listed in _listed_rirs(root):
        path = listed["path"]
        relative = path.relative_to(root)
        kind, source_group = _rir_source_and_group(relative)
        split = _rir_split(source_group)
        metadata = _audio_metadata(path, settings["sample_rate"], require_mono=False)
        for channel in range(metadata["channels"]):
            rows_by_split[split].append({
                "id": "slr28:" + relative.with_suffix("").as_posix() + f":ch{channel}",
                "audio": str(path.resolve()),
                "channel": channel,
                "official_rir_id": listed["rir_id"],
                "room_id": listed["room_id"],
                "rir_kind": kind,
                "source_group": source_group,
                "split": split,
                **metadata,
            })
    if not all(rows_by_split.values()):
        raise ValueError("SLR28 RIR split is empty; inspect source-group inference")
    _audit_group_disjoint(rows_by_split, "source_group")
    report = {"source": "OpenSLR SLR28", "split_method": "held out by database/room family",
              "splits": {}}
    for split, rows in rows_by_split.items():
        manifest = destination / f"{split}.jsonl"
        write_jsonl(manifest, rows)
        report["splits"][split] = {
            "files": len(rows),
            "kinds": dict(Counter(row["rir_kind"] for row in rows)),
            "source_groups": sorted({row["source_group"] for row in rows}),
            "manifest_sha256": sha256_file(manifest),
        }
    write_json(destination / "report.json", report)
    return report


def _audit_group_disjoint(splits, key):
    owners = {}
    for split, rows in splits.items():
        for row in rows:
            group = row[key]
            previous = owners.setdefault(group, split)
            if previous != split:
                raise ValueError(f"Cross-split {key} leakage: {group} in {previous}/{split}")


def build_corruption_manifest(config, split, speech_manifest, noise_manifest, rir_manifest, output):
    """Build the fixed clean/noise/reverb matrix used for dev or synthetic test."""
    if split not in {"dev", "test"}:
        raise ValueError("Fixed corruptions are only valid for dev or test")
    settings = _settings(config)
    speech_rows = read_jsonl(speech_manifest)
    noise_rows = read_jsonl(noise_manifest)
    rir_rows = read_jsonl(rir_manifest)
    if not speech_rows or not noise_rows or not rir_rows:
        raise ValueError("Speech, noise and RIR manifests must be nonempty")
    if any(row["split"] != split for row in noise_rows + rir_rows):
        raise ValueError("Noise/RIR manifest split does not match requested split")
    expected_speech_split = "dev" if split == "dev" else "test_clean"
    if any(row["split"] != expected_speech_split for row in speech_rows):
        raise ValueError("Speech manifest split does not match requested split")
    output_rows = []
    for speech in speech_rows:
        for condition in ("clean", "reverb"):
            seed = _example_seed(config["seed"], split, speech["id"], condition, None)
            rng_index = seed % len(rir_rows)
            output_rows.append({
                "id": f"{speech['id']}:{condition}", "split": split, "condition": condition,
                "speech_id": speech["id"], "speech_audio": speech["audio"], "text": speech["text"],
                "noise_id": None, "noise_audio": None,
                "rir_id": rir_rows[rng_index]["id"] if condition == "reverb" else None,
                "rir_audio": rir_rows[rng_index]["audio"] if condition == "reverb" else None,
                "rir_channel": rir_rows[rng_index].get("channel") if condition == "reverb" else None,
                "snr_db": None, "global_gain_db": 0.0, "noise_crop_start": None, "seed": seed,
            })
        for snr_db in settings["snr_db"]:
            for condition in ("noise", "noise_reverb"):
                seed = _example_seed(config["seed"], split, speech["id"], condition, snr_db)
                noise = noise_rows[seed % len(noise_rows)]
                rir = rir_rows[(seed // len(noise_rows)) % len(rir_rows)]
                output_rows.append({
                    "id": f"{speech['id']}:{condition}:{snr_db}dB", "split": split,
                    "condition": condition, "speech_id": speech["id"],
                    "speech_audio": speech["audio"], "text": speech["text"],
                    "noise_id": noise["id"], "noise_audio": noise["audio"],
                    "rir_id": rir["id"] if condition == "noise_reverb" else None,
                    "rir_audio": rir["audio"] if condition == "noise_reverb" else None,
                    "rir_channel": rir.get("channel") if condition == "noise_reverb" else None,
                    "snr_db": float(snr_db), "global_gain_db": 0.0,
                    "noise_crop_start": seed % noise["frames"], "seed": seed,
                })
    write_jsonl(output, output_rows)
    report = {
        "split": split,
        "examples": len(output_rows),
        "speech_examples": len(speech_rows),
        "conditions": dict(Counter(row["condition"] for row in output_rows)),
        "snr_db": dict(Counter(str(row["snr_db"]) for row in output_rows if row["snr_db"] is not None)),
        "manifest_sha256": sha256_file(output),
    }
    write_json(Path(output).with_suffix(".report.json"), report)
    return report


def _example_seed(seed, split, speech_id, condition, snr_db):
    value = f"{seed}:{split}:{speech_id}:{condition}:{snr_db}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def prepare_all_manifests(config):
    settings = _settings(config)
    root = _resolve(settings["manifest_dir"])
    staging = root.with_name(root.name + ".building")
    if root.exists():
        raise FileExistsError(f"{root} exists; use a new manifest version to preserve fixed splits")
    if staging.exists():
        raise FileExistsError(f"Incomplete staging directory exists: {staging}")
    staging.mkdir(parents=True)
    reports = {
        "speech": build_speech_manifests(config, staging / "speech"),
        "noise": build_noise_manifests(config, staging / "noise"),
        "rir": build_rir_manifests(config, staging / "rir"),
    }
    reports["dev_corruptions"] = build_corruption_manifest(
        config, "dev", staging / "speech/dev.jsonl", staging / "noise/dev.jsonl",
        staging / "rir/dev.jsonl", staging / "corruptions/dev.jsonl"
    )
    reports["test_corruptions"] = build_corruption_manifest(
        config, "test", staging / "speech/test_clean.jsonl", staging / "noise/test.jsonl",
        staging / "rir/test.jsonl", staging / "corruptions/test.jsonl"
    )
    write_json(staging / "report.json", reports)
    staging.replace(root)
    return reports
