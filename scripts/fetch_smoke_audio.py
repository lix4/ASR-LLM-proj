"""Download the official English inference example locally; not a scored benchmark."""

import hashlib
import urllib.request
from pathlib import Path

from earnings_asr.common import read_config, write_json

config = read_config()
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
output = Path(config["data_dir"]) / "smoke" / "asr_en.wav"
output.parent.mkdir(parents=True, exist_ok=True)
if not output.exists():
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"RIFF"):
        raise ValueError("Expected a WAV audio file")
    output.write_bytes(data)
write_json(output.with_suffix(".source.json"), {"url": url,
           "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
           "purpose": "Official smoke example only; no verified reference transcript or WER"})
print(output)
