#!/usr/bin/env python3
"""Select complete utterance groups for a deterministic baseline subset."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from earnings_asr.common import read_jsonl, sha256_file, write_json, write_jsonl


def group_id(row):
    return row.get("speech_id", row["id"])


def select_groups(rows, count, seed):
    identifiers = sorted({group_id(row) for row in rows})
    if count < 1 or count > len(identifiers):
        raise ValueError(f"groups must be in [1, {len(identifiers)}]")
    ranked = sorted(identifiers, key=lambda value: hashlib.sha256(
        f"{seed}:{value}".encode()).digest())
    selected = set(ranked[:count])
    return [row for row in rows if group_id(row) in selected], ranked[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source, output = Path(args.source), Path(args.output)
    if output.exists() or output.with_suffix(".report.json").exists():
        raise FileExistsError("Baseline subset already exists; choose a new output path")
    rows = read_jsonl(source)
    selected, identifiers = select_groups(rows, args.groups, args.seed)
    write_jsonl(output, selected)
    conditions = Counter(row.get("condition", row.get("official_subset", "clean"))
                         for row in selected)
    snrs = Counter(str(row["snr_db"]) for row in selected if row.get("snr_db") is not None)
    write_json(output.with_suffix(".report.json"), {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "selection": "lowest sha256(seed:group_id)",
        "seed": args.seed,
        "groups": len(identifiers),
        "rows": len(selected),
        "conditions": dict(sorted(conditions.items())),
        "snr_db": dict(sorted(snrs.items(), key=lambda item: float(item[0]))),
        "output_sha256": sha256_file(output),
    })


if __name__ == "__main__":
    main()
