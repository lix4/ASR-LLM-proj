#!/usr/bin/env bash
#SBATCH --job-name=qwen-robust-lr2e5
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
python -m earnings_asr.cli \
  --config configs/robust-lr2e-5.json \
  train \
  --train-file data/manifests/robust-v1/sft/train.jsonl \
  --dev-file data/manifests/robust-v1/sft/dev.jsonl \
  --output models/checkpoints/robust-acoustic-lora-r8-lr2e-5-seed42 \
  --augmentation robust \
  --checkpoint models/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0
