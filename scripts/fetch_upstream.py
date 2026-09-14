"""Fetch the exact official source for the SFT adapter, without executing downloaded code."""

import io
import json
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

config = json.loads(Path("configs/default.json").read_text())
revision = config["upstream_revision"]
root = Path("vendor/Qwen3-ASR")
if root.exists() and (root / "REVISION").is_file():
    if (root / "REVISION").read_text().strip() != revision:
        raise RuntimeError("Existing vendor revision differs; inspect before replacing it")
    print(f"Already available: {revision}")
else:
    url = f"https://api.github.com/repos/QwenLM/Qwen3-ASR/tarball/{revision}"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as stream:
        for member in stream.getmembers():
            relative = PurePosixPath(*PurePosixPath(member.name).parts[1:])
            if member.isfile() and not relative.is_absolute() and ".." not in relative.parts:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(stream.extractfile(member).read())
    (root / "REVISION").write_text(revision + "\n")
    print(f"Downloaded official source {revision}; original license retained in {root}/LICENSE")
