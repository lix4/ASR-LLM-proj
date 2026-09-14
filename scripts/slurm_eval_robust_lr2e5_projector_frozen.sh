#!/usr/bin/env bash
#SBATCH --job-name=qwen-eval-pfrozen
#SBATCH --partition=tolga-lab
#SBATCH --exclude=titan,atlas
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/home/xiwenli/ASR-LLM-proj/reports/slurm-%x-%j.out
#SBATCH --error=/home/xiwenli/ASR-LLM-proj/reports/slurm-%x-%j.err

set -euo pipefail

source /home/xiwenli/.bashrc
conda activate py39
cd /home/xiwenli/ASR-LLM-proj

export PYTHONPATH=src

checkpoint_root=models/checkpoints/robust-acoustic-lora-r8-lr2e-5-projector-frozen-seed42
selection_root=outputs/checkpoint-selection/robust-lr2e-5-projector-frozen

for step in 200 400 600 800 889; do
  output_dir="$selection_root/checkpoint-$step"
  mkdir -p "$output_dir"
  if [[ ! -f "$output_dir/summary.json" ]]; then
    echo "START DEV checkpoint-$step"
    python -m earnings_asr.cli \
      --config configs/robust-lr2e-5.json \
      evaluate \
      data/manifests/robust-v1/baseline/dev-100x14.jsonl \
      --output "$output_dir" \
      --checkpoint "$checkpoint_root/checkpoint-$step" \
      > "$output_dir/eval.log" 2>&1
  fi
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("DONE DEV", sys.argv[2], "robust_wer_pct", round(100*d["robust_wer_macro"], 4))' "$output_dir/summary.json" "checkpoint-$step"
done

python - <<'SELECT_PY'
import json
from pathlib import Path

root = Path("outputs/checkpoint-selection/robust-lr2e-5-projector-frozen")
candidates = []
for step in (200, 400, 600, 800, 889):
    path = root / f"checkpoint-{step}" / "summary.json"
    summary = json.loads(path.read_text())
    candidates.append({
        "step": step,
        "checkpoint": f"models/checkpoints/robust-acoustic-lora-r8-lr2e-5-projector-frozen-seed42/checkpoint-{step}",
        "summary": str(path),
        "wer": summary["wer"],
        "robust_wer_macro": summary["robust_wer_macro"],
        "by_condition": summary["by_condition"],
    })
best = min(candidates, key=lambda row: row["robust_wer_macro"])
report = {
    "selection_metric": "dev_generated_robust_wer_macro",
    "experiment": "robust-lr2e-5-projector-frozen",
    "config": "configs/robust-lr2e-5.json",
    "candidates": candidates,
    "selected": best,
}
(root / "selection-summary.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "best-step.txt").write_text(str(best["step"]) + "\n")
print("SELECTED", best["step"], "dev_robust_wer_pct", round(100 * best["robust_wer_macro"], 4))
SELECT_PY

best_step="$(cat "$selection_root/best-step.txt")"
test_output="outputs/final-evaluation/robust-lr2e-5-projector-frozen-checkpoint-${best_step}-test-200x14"
mkdir -p "$test_output"
if [[ ! -f "$test_output/summary.json" ]]; then
  echo "START TEST checkpoint-$best_step"
  python -m earnings_asr.cli \
    --config configs/robust-lr2e-5.json \
    evaluate \
    data/manifests/robust-v1/baseline/test-200x14.jsonl \
    --output "$test_output" \
    --checkpoint "$checkpoint_root/checkpoint-$best_step" \
    --allow-test \
    > "$test_output/eval.log" 2>&1
fi

python - "$best_step" "$test_output" <<'REPORT_PY'
import json
import sys
from pathlib import Path

step = int(sys.argv[1])
test_output = Path(sys.argv[2])
dev = json.loads(Path("outputs/checkpoint-selection/robust-lr2e-5-projector-frozen/selection-summary.json").read_text())
test = json.loads((test_output / "summary.json").read_text())
baseline = json.loads(Path("outputs/baseline-qwen3-asr-0.6b/test-200x14/summary.json").read_text())
report = {
    "experiment": "robust-lr2e-5-projector-frozen",
    "config": "configs/robust-lr2e-5.json",
    "selected_step": step,
    "selected_checkpoint": f"models/checkpoints/robust-acoustic-lora-r8-lr2e-5-projector-frozen-seed42/checkpoint-{step}",
    "dev_selection": dev,
    "test_summary": str(test_output / "summary.json"),
    "test_wer": test["wer"],
    "test_robust_wer_macro": test["robust_wer_macro"],
    "test_by_condition": test["by_condition"],
    "baseline_test_robust_wer_macro": baseline["robust_wer_macro"],
}
Path("outputs/final-evaluation/robust-lr2e-5-projector-frozen-selected.json").write_text(json.dumps(report, indent=2) + "\n")
print("DONE TEST", "checkpoint", step, "robust_wer_pct", round(100 * test["robust_wer_macro"], 4))
REPORT_PY
