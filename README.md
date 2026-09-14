# Robust Qwen3-ASR

This repository evaluates and adapts `Qwen/Qwen3-ASR-0.6B` for English ASR under
additive noise and room reverberation. It uses LibriSpeech speech, MUSAN noise,
held-out SLR28 room impulse responses (RIRs), online acoustic augmentation, and
parameter-efficient supervised fine-tuning (SFT).

The strongest SFT run improves fixed Dev Robust WER from **9.0426% to 8.3506%**, but
does not beat the pretrained model on the held-out Test set: Test Robust WER changes
from **5.8546% to 5.9141%**. This negative result is retained because it shows the
complete checkpoint-selection protocol and the difference between Dev improvement
and Test generalization.

## 🎧 Demo: Projector Frozen fixes a combined noise-and-RIR sample

### Input audio

[![Open the input audio player](docs/assets/examples/play-input-audio.svg)](https://github.com/lix4/ASR-LLM-proj/blob/main/docs/assets/examples/121-127105-0035_noise-reverb_5dB.mp3?raw=1)

**[▶ Play the 14.15-second MP3 input](https://github.com/lix4/ASR-LLM-proj/blob/main/docs/assets/examples/121-127105-0035_noise-reverb_5dB.mp3?raw=1)** · [Download WAV](docs/assets/examples/121-127105-0035_noise-reverb_5dB.wav)

This is a fixed held-out Test input: LibriSpeech target speech is convolved with an
unseen simulated large-room RIR and mixed with interfering MUSAN speech at 5 dB SNR.
Both models receive the **same audio** and decoding settings.

### Output comparison

**Reference**

> SHE PROMISED TO DO THIS AND SHE MENTIONED TO ME THAT WHEN FOR A MOMENT DISBURDENED
> DELIGHTED HE HELD HER HAND THANKING HER FOR THE SACRIFICE SHE ALREADY FELT REWARDED

| Model | Generated output | S / D / I | Sample WER |
| --- | --- | ---: | ---: |
| **Pretrained Qwen3-ASR-0.6B** | “She promised to do this, and she mentioned to me that when for a moment, disburdened, delighted, **she** held her hand, thanking **for** the sacrifice. She already felt rewarded.” | 1 / 1 / 0 | 6.67% |
| **Projector Frozen, checkpoint 100** | “She promised to do this, and she mentioned to me that when for a moment, disburdened, delighted, **he** held her hand, thanking **her** for the sacrifice. She already felt rewarded.” | **0 / 0 / 0** | **0.00%** |

The pretrained output substitutes `she` for `he` and deletes `her`. After the common
scoring normalization, Projector Frozen matches all 30 reference words.

<details>
<summary><strong>Exact input recipe and aggregate context</strong></summary>

| Input field | Value |
| --- | --- |
| Test recipe ID | `121-127105-0035:noise_reverb:5dB` |
| Target speech | LibriSpeech `121-127105-0035`, 14.15 seconds |
| Noise | MUSAN `speech-us-gov-0112`, interfering-speech category |
| RIR | SLR28 `largeroom/Room064/Room064-00053`, channel 0 |
| Target / measured SNR | 5.0 / 5.0 dB |
| Deterministic recipe | seed `7736198490917191944`, noise crop sample `1156360`, peak `0.3197` |

The WAV is a derived evaluation artifact. Target speech is from LibriSpeech (CC BY
4.0), interfering speech is from MUSAN (CC BY 4.0), and the RIR is from OpenSLR SLR28
(Apache 2.0).

This example shows a real sample-level win, not an aggregate Test win. Across all 2,800
Test inputs, Projector Frozen has fewer edit errors on 49 samples, the same count on
2,667, and more errors on 84. Winning samples remove 93 edits; losing samples add 119,
producing the net increase of 26 Test errors reported below.

</details>

The experiment protocol is defined in [`task.md`](task.md). Generated corpora and
model weights live under the `data` and `models` symlinks on shared storage.

## Environment

Use the existing `py39` Conda environment on the cluster:

```bash
conda activate py39
python -m pip install -e '.[dev,service]'
python -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.57.6 accelerate==1.10.1 nagisa==0.2.11 soynlp==0.0.493
python -m pip install --no-deps -e vendor/Qwen3-ASR
```

Qwen3-ASR 0.0.6 declares Python 3.9 support but pins `accelerate==1.12.0`, whose
package requires Python 3.10. This repository uses `accelerate==1.10.1` and carries a
small postponed-annotation compatibility patch in the vendored Transformers backend.

The evaluated model revision is
`5eb144179a02acc5e5ba31e748d22b0cf3e303b0`. GPU runs used one NVIDIA L40S with BF16,
PyTorch 2.8.0, CUDA 12.8, SDPA attention, and the `py39` environment.

## Data and augmentation

Download the exact archives with resume support:

```bash
conda activate py39
PYTHONPATH=src python scripts/fetch_robust_data.py --config configs/robust.json
```

Extract the archives and create immutable versioned manifests:

```bash
PYTHONPATH=src python -m earnings_asr.cli --config configs/robust.json extract-robust-data
PYTHONPATH=src python -m earnings_asr.cli --config configs/robust.json prepare-robust-data
```

The `data/manifests/robust-v1` manifests contain:

- 7,112 LibriSpeech utterances in a deterministic 25-hour training subset;
- 1,638/190/188 MUSAN train/dev/test source recordings, split by complete recording;
- 20,288/23,308/20,214 SLR28 train/dev/test RIR entries, held out by database or
  simulated-room family;
- fixed Dev and Test recipes containing speech, noise, RIR, SNR, crop offset, gain,
  channel, and seed.

Training samples one condition online for every utterance and epoch:

| Condition | Operation | Sampling probability |
| --- | --- | ---: |
| Clean | original speech | 25% |
| Noise | speech + MUSAN segment | 25% |
| Reverb | speech convolved with an SLR28 RIR | 25% |
| Noise + reverb | reverberated speech + MUSAN segment | 25% |

Noise uses `-5`, `0`, `5`, `10`, `15`, or `20` dB SNR. RIR convolution trims leading
silence, normalizes RIR energy, retains its decay shape, and crops the result to the
input length. Noise gain is computed after reverberation. The pipeline records actual
SNR, crop offsets, limiter scale, peak amplitude, and random seed.

Render a fixed example for listening and visual inspection:

```bash
PYTHONPATH=src python scripts/render_augmentation_audit.py --snr 0
```

This writes clean, noise, reverb, and combined WAV files, a concatenated comparison,
measured metadata, and a waveform/spectrogram image to `outputs/augmentation-audit`.

## Evaluation protocol

The same immutable recipes are used for every model:

| Split | Original utterances | Clean | Reverb | Noise | Noise + reverb | Total inputs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 100 | 100 | 100 | 600 | 600 | 1,400 |
| Test | 200 | 200 | 200 | 1,200 | 1,200 | 2,800 |

Noise and noise-plus-reverb contain one input at each of the six SNRs for every
original utterance. The held-out Test noise recordings and RIR families are never used
for training or checkpoint selection.

WER uses corpus-level substitution, deletion, and insertion counts after the same
NFKC, case-folding, punctuation, and whitespace normalization for references and
hypotheses. Two aggregates are reported:

- **Overall WER** pools all errors and reference words. Noise conditions have more
  inputs and therefore more weight.
- **Robust WER** is the unweighted macro-average of Noise, Reverb, and Noise + reverb
  WER. Clean is reported separately and is not included in this selection metric.

Every checkpoint is decoded on fixed Dev inputs. The checkpoint with the lowest
generated Dev Robust WER is selected, and only that checkpoint is decoded on Test.
Teacher-forced validation loss is diagnostic and is not used for model selection.

## Pretrained baseline

The original Qwen3-ASR-0.6B snapshot is loaded directly without SFT.

| Split | Overall WER | Clean | Reverb | Noise | Noise + reverb | Robust WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 10.6539% | 3.2918% | 3.3808% | 11.3138% | 12.4333% | 9.0426% |
| Test | 6.8693% | 1.7675% | 2.1960% | 6.4185% | 8.9493% | **5.8546%** |

The fixed Test run has end-to-end RTF 0.0591, throughput 16.91 audio-seconds per wall
second, P50/P95 request latency of 0.3628/0.8646 seconds, and 1.78 GB peak allocated
GPU memory at batch size one. The complete LibriSpeech `test-other` split has 4.6635%
WER over 2,939 utterances.

The exact baseline reports are
[`Dev summary`](reports/results/baseline-dev-summary.json) and
[`Test summary`](reports/results/baseline-test-summary.json).

## SFT design

All SFT runs use LoRA rank 8, alpha 16, dropout 0.05, batch size 4, gradient
accumulation 2, BF16, seed 42, and LoRA on the 72 `q_proj`, `k_proj`, `v_proj`, and
`out_proj` modules in the 18 audio-encoder self-attention layers. The language-model
decoder, convolutional frontend, embeddings, LM head, and audio-encoder FFNs remain
frozen.

Three projector strategies were evaluated:

| Projector strategy | `proj1` / `proj2` behavior | Trainable parameters | Fraction of model |
| --- | --- | ---: | ---: |
| Direct train | full projector weights train directly | 2,754,432 | 0.3508% |
| Projector LoRA | projector bases frozen; rank-8 LoRA added | 1,061,888 | 0.1355% |
| Projector Frozen | projector weights fully frozen | 1,032,192 | 0.1317% |

“Projector Frozen” means that only the 72 audio-encoder attention LoRA modules are
updated. The `proj1` and `proj2` layers do not receive direct updates or LoRA adapters.

## Experiment matrix

The first `5e-5` comparison trained a clean-only arm and a robust arm for one epoch.
Later experiments retained robust online augmentation, reduced the learning rate to
`2e-5`, and changed only the projector strategy. The final early-checkpoint run stopped
at step 200 and saved steps 50, 100, 150, and 200.

| Experiment | Training data | LR | Projector | Steps | Train time |
| --- | --- | ---: | --- | ---: | ---: |
| Clean-only | 100% clean | 5e-5 | direct train | 889 | 802.2 s |
| Robust SFT | 25% each condition | 5e-5 | direct train | 889 | 949.6 s |
| Robust low-LR | 25% each condition | 2e-5 | direct train | 889 | 809.9 s |
| Robust Projector LoRA | 25% each condition | 2e-5 | LoRA | 889 | 2,601.2 s |
| Robust Projector Frozen | 25% each condition | 2e-5 | frozen | 889 | 1,008.8 s |
| Robust Projector Frozen early | 25% each condition | 2e-5 | frozen | 200 | 463.9 s |

Training diagnostics from the saved reports:

| Experiment | Final train loss | Epoch reached | Steps / second | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Clean-only, 5e-5 | 0.2863 | 1.000 | 1.108 | 6.82 GB |
| Robust, 5e-5 | 0.5084 | 1.000 | 0.936 | 6.82 GB |
| Robust, 2e-5, direct projector | 0.5683 | 1.000 | 1.098 | 6.82 GB |
| Robust, 2e-5, Projector LoRA | 0.8219 | 1.000 | 0.342 | 8.20 GB |
| Robust, 2e-5, Projector Frozen | 0.9783 | 1.000 | 0.881 | 6.80 GB |
| Robust, 2e-5, Projector Frozen early | 1.7113 | 0.225 | 0.431 | 6.61 GB |

Train loss is teacher-forced and is shown as a diagnostic. It is not directly comparable
across clean and dynamically corrupted inputs, and it is not the checkpoint-selection
metric.

### Dev checkpoint curves

The first three full-epoch runs produced the following generated Dev Robust WER:

| Step | Clean-only 5e-5 | Robust 5e-5 | Robust 2e-5 direct |
| ---: | ---: | ---: | ---: |
| 200 | 12.7076% | 15.1863% | 15.3840% |
| 400 | 11.9069% | 12.0527% | 13.6838% |
| 600 | 12.7620% | 11.7759% | 13.8493% |
| 800 | 11.5263% | 11.7462% | **12.3740%** |
| 889 | **11.0617%** | **11.6622%** | 14.5117% |

Changing projector adaptation at `2e-5` moved the best region to the start of
training:

| Step | Projector LoRA | Projector Frozen |
| ---: | ---: | ---: |
| 200 | **9.1736%** | **8.5162%** |
| 400 | 12.5667% | 8.9635% |
| 600 | 12.9943% | 12.9251% |
| 800 | 14.4054% | 13.1376% |
| 889 | 15.6682% | 13.2735% |

The follow-up run resolves the first 200 steps more closely:

| Step | Projector Frozen early Dev Robust WER |
| ---: | ---: |
| 50 | 9.0723% |
| 100 | **8.3506%** |
| 150 | 8.3877% |
| 200 | 8.9166% |

Checkpoint 100 is the final selected SFT checkpoint. It is selected strictly from Dev;
Test is not used for early stopping.

### Selected-checkpoint Test results

| Model | Selected step | Overall | Clean | Reverb | Noise | Noise + reverb | Robust |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained Qwen3-ASR-0.6B | — | **6.8693%** | **1.7675%** | **2.1960%** | 6.4185% | **8.9493%** | **5.8546%** |
| Clean-only, 5e-5, direct projector | 889 | 10.7449% | 4.2046% | 5.2758% | 11.0650% | 12.4264% | 9.5891% |
| Robust, 5e-5, direct projector | 889 | 11.3111% | 4.2582% | 5.3026% | 10.9311% | 13.8681% | 10.0339% |
| Robust, 2e-5, direct projector | 800 | 10.3336% | 4.4992% | 5.5704% | 9.5831% | 12.8504% | 9.3346% |
| Robust, 2e-5, Projector LoRA | 200 | 7.6651% | 2.3567% | 3.1066% | 6.8024% | 10.1723% | 6.6937% |
| Robust, 2e-5, Projector Frozen | 200 | 7.0415% | 1.9550% | 2.3032% | 6.5435% | 9.1769% | 6.0079% |
| **Robust, 2e-5, Projector Frozen early** | **100** | 6.9190% | 1.7943% | 2.2764% | **6.3917%** | 9.0743% | 5.9141% |

Freezing the projector is substantially better than training it directly or adapting
it with LoRA. The early checkpoint is the strongest SFT model, but the pretrained
model remains better on Test Robust WER by 0.0595 percentage points, or 1.02%
relative. Pure Noise is the only main Test condition improved by the selected SFT
checkpoint, by 0.0268 percentage points.

## Detailed pretrained versus best-SFT comparison

In the following tables, delta is `Projector Frozen - Pretrained`; a negative WER
delta is an improvement.

### Main WER and edit counts

| Split | Metric | Pretrained | Projector Frozen | Delta |
| --- | --- | ---: | ---: | ---: |
| Dev | Overall WER | 10.6539% | **9.8087%** | **-0.8452 pp** |
| Dev | Robust WER | 9.0426% | **8.3506%** | **-0.6920 pp** |
| Dev | Clean WER | 3.2918% | **3.2473%** | -0.0445 pp |
| Dev | Reverb WER | 3.3808% | **3.2473%** | -0.1335 pp |
| Dev | Noise WER | 11.3138% | **9.4454%** | **-1.8683 pp** |
| Dev | Noise + reverb WER | 12.4333% | **12.3591%** | -0.0741 pp |
| Test | Overall WER | **6.8693%** | 6.9190% | +0.0497 pp |
| Test | Robust WER | **5.8546%** | 5.9141% | +0.0595 pp |
| Test | Clean WER | **1.7675%** | 1.7943% | +0.0268 pp |
| Test | Reverb WER | **2.1960%** | 2.2764% | +0.0803 pp |
| Test | Noise WER | 6.4185% | **6.3917%** | **-0.0268 pp** |
| Test | Noise + reverb WER | **8.9493%** | 9.0743% | +0.1250 pp |

| Split | Model | Substitutions | Deletions | Insertions | Reference words | Total edits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dev | Pretrained | 2,236 | 518 | 599 | 31,472 | 3,353 |
| Dev | Projector Frozen | 2,214 | 523 | 350 | 31,472 | **3,087** |
| Test | Pretrained | 2,582 | 601 | 408 | 52,276 | **3,591** |
| Test | Projector Frozen | 2,567 | 633 | 417 | 52,276 | 3,617 |

The Dev gain is dominated by fewer insertions and by the 0 dB Noise bucket. That
specific improvement does not reproduce on Test, which explains the mismatch between
Dev checkpoint selection and final Test performance.

### WER by SNR

| Split | SNR | Noise pretrained | Noise frozen | Delta | Noise + reverb pretrained | Noise + reverb frozen | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | -5 dB | 24.8665% | **24.6886%** | -0.1779 pp | 33.9413% | **33.3630%** | -0.5783 pp |
| Dev | 0 dB | 25.7117% | **14.5463%** | **-11.1655 pp** | 18.1050% | **18.0605%** | -0.0445 pp |
| Dev | 5 dB | **6.2278%** | 6.4947% | +0.2669 pp | 9.7865% | 9.7865% | 0.0000 pp |
| Dev | 10 dB | 4.0480% | 4.0480% | 0.0000 pp | **4.9377%** | 5.0712% | +0.1335 pp |
| Dev | 15 dB | 3.5587% | **3.5142%** | -0.0445 pp | 4.0480% | **4.0036%** | -0.0445 pp |
| Dev | 20 dB | 3.4698% | **3.3808%** | -0.0890 pp | **3.7811%** | 3.8701% | +0.0890 pp |
| Test | -5 dB | 20.0054% | **19.6572%** | -0.3482 pp | **25.6294%** | 25.9775% | +0.3482 pp |
| Test | 0 dB | **8.7574%** | 8.9181% | +0.1607 pp | 14.5956% | **14.4617%** | -0.1339 pp |
| Test | 5 dB | **4.0171%** | 4.0707% | +0.0536 pp | **5.0884%** | 5.2223% | +0.1339 pp |
| Test | 10 dB | **1.8747%** | 1.9014% | +0.0268 pp | **3.6154%** | 3.7225% | +0.1071 pp |
| Test | 15 dB | **1.9550%** | 1.9818% | +0.0268 pp | **2.5442%** | 2.7317% | +0.1875 pp |
| Test | 20 dB | 1.9014% | **1.8211%** | -0.0803 pp | **2.2228%** | 2.3299% | +0.1071 pp |

### WER by noise category

Noise-category metrics pool Noise and Noise + reverb examples that use the given
MUSAN category.

| Split | Category | Pretrained | Projector Frozen | Delta |
| --- | --- | ---: | ---: | ---: |
| Dev | Music | 6.3459% | **6.2549%** | -0.0910 pp |
| Dev | Environmental noise | **6.8236%** | 6.8759% | +0.0523 pp |
| Dev | Interfering speech | 30.5113% | **26.0752%** | **-4.4362 pp** |
| Test | Music | **6.3600%** | 6.3902% | +0.0301 pp |
| Test | Environmental noise | **3.9333%** | 4.0565% | +0.1232 pp |
| Test | Interfering speech | 21.8882% | **21.7617%** | -0.1264 pp |

### Inference measurements

These are end-to-end local measurements at batch size one and concurrency one. They
include audio decode, resampling, segmentation, encoding, and generation.

| Split | Model | RTF | Audio seconds / wall second | P50 latency | P95 latency |
| --- | --- | ---: | ---: | ---: | ---: |
| Dev | Pretrained | 0.06362 | 15.72 | 0.3984 s | 1.1786 s |
| Dev | Projector Frozen | 0.05895 | 16.96 | 0.3613 s | 1.1094 s |
| Test | Pretrained | 0.05915 | 16.91 | 0.3628 s | 0.8646 s |
| Test | Projector Frozen | 0.05568 | 17.96 | 0.3302 s | 0.8444 s |

These timing differences are descriptive single-run measurements; they are not a
controlled performance benchmark with repeated trials.

## Reproducing training and evaluation

The final early-checkpoint experiment is submitted with the cluster resource policy in
`scripts/slurm_train_eval_robust_lr2e5_projector_frozen_early.sh`:

```bash
sbatch --partition=tolga-lab --exclude=titan,atlas --gres=gpu:1 \
  --cpus-per-task=8 --mem=40G --time=10:00:00 \
  scripts/slurm_train_eval_robust_lr2e5_projector_frozen_early.sh
```

The script trains to step 200, saves steps 50/100/150/200, decodes every checkpoint on
Dev, selects the lowest Dev Robust WER, and evaluates that checkpoint once on Test.

Key machine-readable artifacts:

- [`early checkpoint selection`](reports/results/best-selection-summary.json)
- [`selected experiment record`](reports/results/best-experiment.json)
- [`selected Dev details`](reports/results/best-dev-summary.json)
- [`selected Test details`](reports/results/best-test-summary.json)
- [`paired example record`](reports/results/paired-example.json)
- [`all selected experiment Test details`](reports/results/best-test-summary.json)

## Verification

```bash
PYTHONPATH=src python -m pytest -q
python -m ruff check src tests scripts
```

The augmentation tests cover deterministic reproduction, known SNR, identity RIR,
output length, finite values, and clipping. Training smoke tests verify LoRA target
scope, gradients, frozen parameters, adapter save/reload, and projector mode.

The first verified GPU smoke test used a 5.855-second LibriSpeech sample and produced a
correct transcription with end-to-end RTF 0.511. Its machine-readable result is
`outputs/baseline-py39-gpu-smoke.json`; the recorded environment is
`reports/environment-py39-l40s.json`.
