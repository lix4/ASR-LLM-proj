"""Pinned official qwen-asr backend shared by CLI, evaluation and service."""

import json
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from .audio import SAMPLE_RATE, load_audio, segment


def resolve_model(config, checkpoint=None):
    if checkpoint:
        path = Path(checkpoint).resolve()
        if not (path / "config.json").is_file():
            raise ValueError("Checkpoint must be a local, complete Hugging Face model directory")
        return str(path)
    # Download first: upstream forwards revision to the model but not its processor.
    # Loading both from this exact local snapshot prevents revision drift.
    return snapshot_download(config["model_id"], revision=config["model_revision"],
                             cache_dir=str(Path(config["model_dir"]) / "hub"),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])


def resolve_inference_artifacts(config, checkpoint=None):
    """Return the complete base model path and an optional PEFT adapter path."""
    if checkpoint and (Path(checkpoint) / "adapter_config.json").is_file():
        adapter_path = Path(checkpoint).resolve()
        adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
        base_path = Path(adapter_config.get("base_model_name_or_path", "")).resolve()
        if not (base_path / "config.json").is_file():
            raise ValueError("Adapter does not reference a complete local base model")
        return str(base_path), str(adapter_path)
    return resolve_model(config, checkpoint), None


class QwenBackend:
    def __init__(self, config, checkpoint=None):
        import torch
        from qwen_asr import Qwen3ASRModel
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        self.config = config
        self.device = config["device"]
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if self.device == "cpu":
            torch.set_num_threads(config["cpu_threads"])
        dtype = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_bf16_supported() else (
            torch.float16 if self.device.startswith("cuda") else torch.float32)
        if config["batch_size"] < 1:
            raise ValueError("batch_size must be at least 1")
        self.path, self.adapter_path = resolve_inference_artifacts(config, checkpoint)
        start = time.perf_counter()
        self.wrapper = Qwen3ASRModel.from_pretrained(
            self.path, dtype=dtype, device_map=self.device, attn_implementation="sdpa",
            max_inference_batch_size=config["batch_size"], max_new_tokens=config["max_new_tokens"])
        if self.adapter_path:
            from peft import PeftModel

            adapter = PeftModel.from_pretrained(self.wrapper.model, self.adapter_path)
            self.wrapper.model = adapter.merge_and_unload(safe_merge=True)
        self.wrapper.model.eval()
        self.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.metadata = {"model_id": config["model_id"],
                         "model_revision": config["model_revision"],
                         "checkpoint": str(checkpoint) if checkpoint else None,
                         "base_checkpoint": self.path,
                         "adapter": self.adapter_path,
                         "device": self.device, "dtype": str(dtype), "attention": "sdpa",
                         "batch_size": config["batch_size"], "max_new_tokens": config["max_new_tokens"],
                         "language": "English", "context": "", "do_sample": False,
                         "chunk_seconds": config["chunk_seconds"],
                         "boundary_search_seconds": config["boundary_search_seconds"],
                         "silence_rms": config["silence_rms"], "cpu_threads": config["cpu_threads"]}

    def synchronize(self):
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize(self.device)

    def reset_peak(self):
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_bytes(self):
        if self.device.startswith("cuda"):
            import torch
            return torch.cuda.max_memory_allocated(self.device)
        return None

    def transcribe_batch(self, arrays):
        import torch
        with torch.inference_mode():
            results = self.wrapper.transcribe(audio=[(a, SAMPLE_RATE) for a in arrays], language="English")
        if len(results) != len(arrays):
            raise RuntimeError("Model returned the wrong number of results")
        return [result.text for result in results]


def transcribe_file(backend, path, config, max_seconds=None):
    start = time.perf_counter()
    audio = load_audio(path, max_seconds=max_seconds)
    decode_seconds = time.perf_counter() - start
    result = transcribe_waveform(backend, audio, config)
    result["processing_seconds"] += decode_seconds
    result["preprocessing_seconds"] += decode_seconds
    result["rtf"] = result["processing_seconds"] / result["duration_seconds"]
    return result


def transcribe_waveform(backend, audio, config):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio):
        raise ValueError("Audio must be a nonempty mono waveform")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    backend.synchronize()
    start = time.perf_counter()
    chunks = list(segment(audio, config["chunk_seconds"], config["boundary_search_seconds"],
                          config["silence_rms"]))
    preprocessing = time.perf_counter() - start
    segments = [{"start": c.start / SAMPLE_RATE, "end": c.end / SAMPLE_RATE,
                 "text": "", "silent": c.silent} for c in chunks]
    active = [i for i, c in enumerate(chunks) if not c.silent]
    model_start = time.perf_counter()
    for offset in range(0, len(active), config["batch_size"]):
        indices = active[offset:offset + config["batch_size"]]
        predictions = backend.transcribe_batch([audio[chunks[i].start:chunks[i].end] for i in indices])
        for i, prediction in zip(indices, predictions):
            segments[i]["text"] = prediction.strip()
    backend.synchronize()
    inference = time.perf_counter() - model_start
    text = " ".join(s["text"] for s in segments if s["text"])
    elapsed = time.perf_counter() - start
    duration = len(audio) / SAMPLE_RATE
    return {"text": text, "segments": segments, "duration_seconds": duration,
            "processing_seconds": elapsed, "preprocessing_seconds": preprocessing,
            "inference_seconds": inference, "rtf": elapsed / duration,
            "timestamps": "audio_segment_boundaries", "segmentation": "silence_aware_nonoverlapping"}


def latency_summary(seconds):
    return {"p50_seconds": float(np.percentile(seconds, 50)),
            "p95_seconds": float(np.percentile(seconds, 95))}
