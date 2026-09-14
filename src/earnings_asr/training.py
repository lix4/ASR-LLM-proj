"""Small adapter around the pinned official SFT implementation.

Reuse the official forward patch, preprocessing, Trainer and checkpoint callback.
Correct label masking for left padding, use a local model snapshot, and evaluate on dev.
"""

import importlib.util
import math
import time
from pathlib import Path

import torch

from .audio import load_audio
from .common import read_jsonl, sha256_file, write_json
from .inference import resolve_model
from .online_data import OnlineWaveformAugmenter


def load_official(config):
    root = Path("vendor/Qwen3-ASR")
    if not (root / "REVISION").is_file() or (root / "REVISION").read_text().strip() != config["upstream_revision"]:
        raise RuntimeError("Run scripts/fetch_upstream.py to obtain the pinned official SFT implementation")
    spec = importlib.util.spec_from_file_location("qwen_official_sft", root / "finetuning/qwen3_asr_sft.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mask_target_labels(full, prefix):
    """Mask prompt/audio/padding using positions among nonpadding tokens, on either side."""
    labels = full["input_ids"].clone()
    for i in range(labels.shape[0]):
        full_valid = full["attention_mask"][i].bool()
        prefix_ids = prefix["input_ids"][i][prefix["attention_mask"][i].bool()]
        full_ids = full["input_ids"][i][full_valid]
        if not torch.equal(full_ids[:len(prefix_ids)], prefix_ids):
            raise ValueError("Tokenized prefix is not an exact prefix of the full training sequence")
        if len(full_ids) <= len(prefix_ids):
            raise ValueError("Training sample has no target tokens")
        positions = full["attention_mask"][i].long().cumsum(0)
        labels[i][(~full_valid) | (positions <= len(prefix_ids))] = -100
    return labels


class TargetOnlyCollator:
    def __init__(self, processor, augmenter=None, audit_limit=16):
        self.processor = processor
        self.augmenter = augmenter
        self.audit_limit = audit_limit
        self.augmentation_audit = []

    def set_epoch(self, epoch):
        if self.augmenter is not None:
            self.augmenter.set_epoch(epoch)

    def __call__(self, features):
        audios = []
        for feature in features:
            audio = load_audio(feature["audio"])
            if self.augmenter is not None and feature.get("split") == "train":
                sample_id = feature.get("id", feature["audio"])
                audio, metadata = self.augmenter(audio, sample_id)
                if len(self.augmentation_audit) < self.audit_limit:
                    self.augmentation_audit.append(metadata)
            audios.append(audio)
        prefixes = [f["prefix_text"] for f in features]
        eos = self.processor.tokenizer.eos_token
        if not eos:
            raise ValueError("Tokenizer must define an EOS token")
        full = self.processor(text=[p + f["target"] + eos for p, f in zip(prefixes, features)],
                              audio=audios, return_tensors="pt", padding=True, truncation=False)
        prefix = self.processor(text=prefixes, audio=audios, return_tensors="pt", padding=True,
                                truncation=False)
        full["labels"] = mask_target_labels(full, prefix)
        return full


def acoustic_lora_targets(model):
    suffixes = (".self_attn.q_proj", ".self_attn.k_proj",
                ".self_attn.v_proj", ".self_attn.out_proj")
    targets = [name for name, _ in model.named_modules()
               if name.startswith("thinker.audio_tower.layers.") and name.endswith(suffixes)]
    if not targets or len(targets) % len(suffixes):
        raise RuntimeError("Could not identify complete audio-attention LoRA targets")
    return targets


def apply_acoustic_lora(model, settings):
    from peft import LoraConfig, get_peft_model

    attention_targets = acoustic_lora_targets(model)
    projectors = ("thinker.audio_tower.proj1", "thinker.audio_tower.proj2")
    module_names = {name for name, _ in model.named_modules()}
    if not set(projectors).issubset(module_names):
        raise RuntimeError("Could not identify the audio-to-language projector")
    projector_mode = settings.get("projector_mode", "train")
    if projector_mode not in {"train", "frozen", "lora"}:
        raise ValueError("projector_mode must be train, frozen, or lora")
    targets = list(attention_targets)
    modules_to_save = None
    if projector_mode == "train":
        modules_to_save = list(projectors)
    elif projector_mode == "lora":
        targets.extend(projectors)
    adapter = get_peft_model(model, LoraConfig(
        r=settings["lora_r"],
        lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"],
        bias="none",
        target_modules=targets,
        modules_to_save=modules_to_save,
    ))
    trainable = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]
    unexpected = [name for name in trainable
                  if ".lora_" not in name and ".modules_to_save." not in name]
    decoder_trainable = [name for name in trainable
                         if ".thinker.model." in name or ".thinker.lm_head." in name]
    if unexpected or decoder_trainable:
        raise RuntimeError("LoRA freezing audit found unexpected trainable parameters")
    counts = {
        "total": sum(parameter.numel() for parameter in adapter.parameters()),
        "trainable": sum(parameter.numel() for parameter in adapter.parameters()
                         if parameter.requires_grad),
    }
    counts["trainable_fraction"] = counts["trainable"] / counts["total"]
    return adapter, {
        "target_modules": targets,
        "attention_lora_target_modules": attention_targets,
        "projector_modules": list(projectors),
        "projector_mode": projector_mode,
        "trainable_parameter_names": trainable,
        "parameter_counts": counts,
        "decoder_frozen": not decoder_trainable,
    }


def train(config, train_file, dev_file, output, smoke=False, max_steps=-1, seed=None,
          augmentation_mode="clean", checkpoint=None, projector_mode=None,
          checkpoint_steps=None):
    from datasets import load_dataset
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig, TrainerCallback, TrainingArguments, set_seed

    seed = config["seed"] if seed is None else seed
    set_seed(seed)
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("Domain training requires a CUDA machine; use --smoke only for CPU plumbing checks")
    if Path(output).exists():
        raise FileExistsError("Use a new training output directory")
    if not read_jsonl(train_file) or not read_jsonl(dev_file):
        raise ValueError("Both train and dev JSONL files must be nonempty")
    if augmentation_mode not in {"clean", "robust"}:
        raise ValueError("augmentation_mode must be clean or robust")
    if checkpoint_steps is not None and checkpoint_steps < 1:
        raise ValueError("checkpoint_steps must be positive")
    official = load_official(config)
    model_path = resolve_model(config, checkpoint)
    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)
    if not use_cuda:
        torch.set_num_threads(config["cpu_threads"])
    wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=dtype, device_map=None,
                                           attn_implementation="sdpa")
    model, processor = wrapper.model, wrapper.processor
    official.patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    training_settings = dict(config["training"])
    if projector_mode is not None:
        training_settings["projector_mode"] = projector_mode
    model, lora_report = apply_acoustic_lora(model, training_settings)
    raw = load_dataset("json", data_files={"train": train_file, "validation": dev_file})
    ds = raw.map(official.make_preprocess_fn_prefix_only(processor))
    augmenter = None
    if augmentation_mode == "robust":
        manifest_root = Path(config["robust_data"]["manifest_dir"])
        augmenter = OnlineWaveformAugmenter.from_manifests(
            config, manifest_root / "noise/train.jsonl", manifest_root / "rir/train.jsonl")
    collator = TargetOnlyCollator(processor, augmenter)
    inspected = collator([ds["train"][i] for i in range(min(2, len(ds["train"])))])

    class TrainingAuditCallback(TrainerCallback):
        def __init__(self, data_collator):
            self.data_collator = data_collator
            self.gradient_audit = None

        def on_epoch_begin(self, args, state, control, **kwargs):
            epoch = int(math.floor(state.epoch or 0.0))
            self.data_collator.set_epoch(epoch)
            return control

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if self.gradient_audit is None:
                with_gradient, nonfinite, frozen_with_gradient = [], [], []
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        continue
                    if parameter.requires_grad:
                        with_gradient.append(name)
                        if not torch.isfinite(parameter.grad).all():
                            nonfinite.append(name)
                    else:
                        frozen_with_gradient.append(name)
                self.gradient_audit = {
                    "trainable_with_gradient": with_gradient,
                    "nonfinite_gradients": nonfinite,
                    "frozen_with_gradient": frozen_with_gradient,
                }
            return control

    audit_callback = TrainingAuditCallback(collator)
    periodic_steps = 1 if smoke else (checkpoint_steps or 200)
    settings = dict(output_dir=str(output),
                    per_device_train_batch_size=training_settings["per_device_train_batch_size"],
                    per_device_eval_batch_size=training_settings["per_device_eval_batch_size"],
                    gradient_accumulation_steps=1 if smoke else training_settings["gradient_accumulation_steps"],
                    learning_rate=training_settings["learning_rate"], num_train_epochs=1,
                    max_steps=1 if smoke else max_steps,
                    logging_steps=1 if smoke else 10, save_strategy="steps", save_steps=periodic_steps,
                    eval_strategy="steps", eval_steps=periodic_steps,
                    save_total_limit=2 if smoke else 5,
                    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
                    bf16=bf16, fp16=use_cuda and not bf16, use_cpu=not use_cuda,
                    remove_unused_columns=False, report_to="none", seed=seed, data_seed=seed,
                    dataloader_num_workers=0, dataloader_pin_memory=use_cuda,
                    optim="adamw_torch", warmup_ratio=0.02,
                    length_column_name="frames", group_by_length=not smoke)
    trainer = official.CastFloatInputsTrainer(
        model=model, args=TrainingArguments(**settings), train_dataset=ds["train"],
        eval_dataset=ds["validation"], data_collator=collator,
        processing_class=processor,
        callbacks=[audit_callback])
    start = time.perf_counter()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    trainer.save_model(str(Path(output) / "final"))
    processor.save_pretrained(str(Path(output) / "final"))
    report = {"smoke_only": smoke, "seed": seed, "augmentation_mode": augmentation_mode,
              **lora_report,
              "model_revision": config["model_revision"], "upstream_revision": config["upstream_revision"],
              "train_file_sha256": sha256_file(train_file), "dev_file_sha256": sha256_file(dev_file),
              "label_target_token_counts": (inspected["labels"] != -100).sum(1).tolist(),
              "training_seconds": time.perf_counter() - start, "metrics": result.metrics,
              "best_checkpoint": trainer.state.best_model_checkpoint,
              "selection_metric": "dev loss for plumbing only; generated Robust WER selects full runs",
              "gradient_audit": audit_callback.gradient_audit,
              "augmentation_audit": collator.augmentation_audit,
              "settings": settings,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if use_cuda else None}
    write_json(Path(output) / "training-report.json", report)
    return report
