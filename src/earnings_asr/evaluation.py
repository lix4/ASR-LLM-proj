"""Fixed-manifest evaluation with per-sample evidence and explicit timing scope."""

import json
import time
from pathlib import Path

from .audio import load_audio
from .augmentation import augment_waveform
from .common import read_jsonl, sha256_file, write_json
from .inference import QwenBackend, latency_summary, transcribe_waveform
from .online_data import load_rir_channel
from .scoring import aggregate, score_pair


def _input_paths(row):
    if row.get("audio"):
        return [row["audio"]]
    paths = [row.get("speech_audio")]
    if row.get("noise_audio"):
        paths.append(row["noise_audio"])
    if row.get("rir_audio"):
        paths.append(row["rir_audio"])
    if not paths[0]:
        raise ValueError(f"Manifest row {row.get('id')} has no audio input")
    return paths


def render_manifest_row(row, settings):
    """Load a legacy clean row or deterministically render a corruption recipe."""
    if row.get("audio"):
        return load_audio(row["audio"]), {"condition": row.get("condition", "clean")}
    speech = load_audio(row["speech_audio"])
    noise = load_audio(row["noise_audio"]) if row.get("noise_audio") else None
    rir = (load_rir_channel(row["rir_audio"], row.get("rir_channel", 0))
           if row.get("rir_audio") else None)
    return augment_waveform(
        speech,
        row["condition"],
        noise=noise,
        rir=rir,
        snr_db=row.get("snr_db"),
        noise_crop_start=row.get("noise_crop_start") or 0,
        global_gain_db=row.get("global_gain_db", 0.0),
        tail_policy=settings["rir_tail_policy"],
        peak_ceiling=settings["peak_ceiling"],
    )


def _condition(row):
    return row.get("condition") or row.get("official_subset") or "clean"


def _noise_category(row):
    noise_id = row.get("noise_id")
    if not noise_id or ":" not in noise_id:
        return None
    return noise_id.split(":", 1)[1].split("/", 1)[0]


def _grouped_metrics(rows, key):
    values = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.setdefault(str(value), []).append(row)
    return {value: aggregate(group) for value, group in sorted(values.items())}


def _quality_summary(predictions):
    by_condition = _grouped_metrics(predictions, "condition")
    degraded = [by_condition[name]["wer"] for name in ("noise", "reverb", "noise_reverb")
                if name in by_condition and by_condition[name]["wer"] is not None]
    by_condition_snr = {}
    for condition in ("noise", "noise_reverb"):
        selected = [row for row in predictions if row["condition"] == condition]
        if selected:
            by_condition_snr[condition] = _grouped_metrics(selected, "snr_db")
    return {
        "by_condition": by_condition,
        "by_condition_and_snr_db": by_condition_snr,
        "by_noise_category": _grouped_metrics(predictions, "noise_category"),
        "robust_wer_macro": sum(degraded) / len(degraded) if degraded else None,
        "robust_wer_conditions": [name for name in ("noise", "reverb", "noise_reverb")
                                  if name in by_condition],
    }


def evaluate(config, manifest, output, checkpoint=None, allow_test=False):
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError("Cannot evaluate an empty manifest")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Manifest contains duplicate IDs")
    if any(row["split"].startswith("test") for row in rows) and not allow_test:
        raise ValueError("Fixed test evaluation requires --allow-test; use dev while debugging")
    if any(row["split"] not in {"train", "dev", "test", "test_clean", "test_other", "smoke"}
           for row in rows):
        raise ValueError("Unknown manifest split")
    for row in rows:
        for path in _input_paths(row):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    output = Path(output)
    partial_path = output / "predictions.partial.jsonl"
    final_path = output / "predictions.jsonl"
    summary_path = output / "summary.json"
    if summary_path.exists() or final_path.exists():
        raise FileExistsError("Evaluation output is already complete; use a new output directory")
    predictions = read_jsonl(partial_path) if partial_path.exists() else []
    if predictions and [row["id"] for row in predictions] != [row["id"] for row in rows[:len(predictions)]]:
        raise ValueError("Partial predictions do not match the manifest prefix")
    if len(predictions) > len(rows):
        raise ValueError("Partial predictions contain more rows than the manifest")
    backend = QwenBackend(config, checkpoint)
    cold_audio, _ = render_manifest_row(rows[0], config["robust_data"])
    cold = transcribe_waveform(backend, cold_audio, config)
    backend.reset_peak()
    output.mkdir(parents=True, exist_ok=True)
    resumed_samples = len(predictions)
    start = time.perf_counter()
    try:
        mode = "a" if predictions else "w"
        with partial_path.open(mode, encoding="utf-8") as stream:
            for row in rows[len(predictions):]:
                render_start = time.perf_counter()
                audio, augmentation = render_manifest_row(row, config["robust_data"])
                augmentation_seconds = time.perf_counter() - render_start
                result = transcribe_waveform(backend, audio, config)
                result["augmentation_seconds"] = augmentation_seconds
                result["processing_seconds"] += augmentation_seconds
                result["preprocessing_seconds"] += augmentation_seconds
                result["rtf"] = result["processing_seconds"] / result["duration_seconds"]
                condition = _condition(row)
                prediction = {
                    "id": row["id"], "speech_id": row.get("speech_id", row["id"]),
                    "condition": condition, "snr_db": row.get("snr_db"),
                    "noise_id": row.get("noise_id"), "noise_category": _noise_category(row),
                    "rir_id": row.get("rir_id"), "reference": row["text"],
                    "prediction": result["text"], "augmentation": augmentation,
                    **score_pair(row["text"], result["text"]), **result,
                }
                predictions.append(prediction)
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                stream.flush()
                if len(predictions) % 100 == 0 or len(predictions) == len(rows):
                    done_seconds = sum(item["duration_seconds"] for item in predictions[resumed_samples:])
                    wall_seconds = time.perf_counter() - start
                    rtf = wall_seconds / done_seconds if done_seconds else None
                    print(f"evaluation progress: {len(predictions)}/{len(rows)}, rtf={rtf:.4f}",
                          flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"completed_samples": len(predictions),
                                             "error_type": type(error).__name__,
                                             "error": str(error)})
        raise
    elapsed = time.perf_counter() - start
    summary = {**aggregate(predictions), **_quality_summary(predictions), "model": backend.metadata,
               "manifest_sha256": sha256_file(manifest), "splits": sorted({r["split"] for r in rows}),
               "model_load_seconds_excluding_download": backend.load_seconds,
               "cold_first_request_seconds": cold["processing_seconds"],
               "warmup_requests": 1, "concurrency": 1, "evaluation_wall_seconds": elapsed,
               "resumed_samples": resumed_samples,
               "audio_seconds": sum(p["duration_seconds"] for p in predictions),
               "peak_gpu_allocated_bytes": backend.peak_bytes(),
               "latency": latency_summary([p["processing_seconds"] for p in predictions]),
               "input_duration": latency_summary([p["duration_seconds"] for p in predictions]),
               "timing_scope": "local requests: decode, resample, segment, encode and generate; no HTTP/queue",
               "limitations": ["Encoder/generation timing not separated in this evaluation entrypoint."]}
    summary["rtf_end_to_end"] = elapsed / summary["audio_seconds"]
    summary["audio_seconds_per_wall_second"] = summary["audio_seconds"] / elapsed
    partial_path.replace(final_path)
    failure_path = output / "failure.json"
    if failure_path.exists():
        failure_path.unlink()
    write_json(summary_path, summary)
    return summary
