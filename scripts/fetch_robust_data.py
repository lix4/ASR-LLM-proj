#!/usr/bin/env python3
"""Download the exact first-stage OpenSLR archives with wget resume support."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/robust.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    settings = config["robust_data"]
    destination = Path(settings["raw_dir"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "--continue", f"--directory-prefix={destination}", *settings["archives"].values(), *settings.get("checksum_files", {}).values()],
        check=True,
    )


if __name__ == "__main__":
    main()
