"""Read-only environment and gated-data diagnostics; never record credentials."""

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_url


def inspect_environment(config, network=True):
    import torch

    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpu.append({"index": i, "name": torch.cuda.get_device_name(i),
                        "free_bytes": free, "total_bytes": total})
    packages = {}
    for package in ("torch", "torchaudio", "transformers", "qwen-asr", "datasets",
                    "numpy", "soundfile", "jiwer", "pyarrow", "fastapi", "pillow", "peft"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    disk = shutil.disk_usage(Path.cwd())
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version,
        "cpu_count": os.cpu_count(), "packages": packages,
        "cuda_available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda,
        "gpus": gpu, "disk_free_bytes": disk.free,
        "disk_note": "Filesystem availability is not an individual user quota.",
        "tools": {key: bool(shutil.which(key)) for key in ("ffmpeg", "nvidia-smi", "nvcc")},
        "hf_token_present": bool(get_token()),
    }
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        report["nvidia_smi"] = result.stdout.strip()
    if network:
        api = HfApi()
        repositories = [("model", config["model_id"], config["model_revision"])]
        if config.get("dataset_id"):
            repositories.append(("dataset", config["dataset_id"], config["dataset_revision"]))
        for kind, repo, revision in repositories:
            try:
                info = api.repo_info(repo, repo_type=kind, revision=revision, timeout=20)
                report[kind] = {"reachable": True, "revision": info.sha, "gated": info.gated}
            except Exception as error:
                report[kind] = {"reachable": False, "error_type": type(error).__name__}
        if config.get("dataset_id"):
            try:
                get_hf_file_metadata(hf_hub_url(config["dataset_id"], "S/train-00000-of-00006.parquet",
                                              repo_type="dataset", revision=config["dataset_revision"]),
                                     token=get_token(), timeout=20)
                report["spgispeech_access"] = "granted"
            except Exception as error:
                report["spgispeech_access"] = "unavailable"
                report["spgispeech_access_error"] = type(error).__name__
        if config.get("robust_data"):
            report["openslr_archives"] = list(config["robust_data"]["archives"].values())
    return report
