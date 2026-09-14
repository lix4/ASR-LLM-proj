"""Epoch-aware online acoustic augmentation used by the future training dataset."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np
import soundfile as sf

from .audio import SAMPLE_RATE, load_audio
from .augmentation import OnlineAugmentationSampler, augment_waveform
from .common import read_jsonl


def load_rir_channel(path, channel):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or channel is None or not 0 <= channel < audio.shape[1]:
        raise ValueError(f"Invalid RIR channel request: {path} channel {channel}")
    return audio[:, channel]


class OnlineWaveformAugmenter:
    """Load held-in resources and resample a reproducible recipe on every epoch."""

    def __init__(self, settings, noise_rows, rir_rows, seed=42, loader=load_audio,
                 rir_loader=load_rir_channel):
        if not noise_rows or not rir_rows:
            raise ValueError("Online augmentation requires nonempty noise and RIR pools")
        if any(row.get("split") != "train" for row in noise_rows + rir_rows):
            raise ValueError("Online training pools may only contain train resources")
        self.settings = settings
        self.noise_rows = tuple(noise_rows)
        self.rir_rows = tuple(rir_rows)
        self.seed = int(seed)
        self.loader = loader
        self.rir_loader = rir_loader
        self.epoch = 0
        self.sampler = OnlineAugmentationSampler(
            settings["condition_probabilities"], settings["snr_db"],
            settings["global_gain_db"], seed
        )

    @classmethod
    def from_manifests(cls, config, noise_manifest, rir_manifest, loader=load_audio,
                       rir_loader=load_rir_channel):
        return cls(
            config["robust_data"], read_jsonl(noise_manifest), read_jsonl(rir_manifest),
            config["seed"], loader, rir_loader
        )

    def set_epoch(self, epoch):
        if int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = int(epoch)

    def recipe(self, sample_id):
        digest = hashlib.sha256(f"{self.seed}:{self.epoch}:{sample_id}:resources".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        noise = self.noise_rows[int(rng.integers(len(self.noise_rows)))]
        rir = self.rir_rows[int(rng.integers(len(self.rir_rows)))]
        choice = self.sampler.sample(sample_id, self.epoch, noise["frames"])
        noisy = choice.condition in {"noise", "noise_reverb"}
        reverberant = choice.condition in {"reverb", "noise_reverb"}
        return {
            **asdict(choice),
            "sample_id": sample_id,
            "epoch": self.epoch,
            "noise_id": noise["id"] if noisy else None,
            "noise_audio": noise["audio"] if noisy else None,
            "rir_id": rir["id"] if reverberant else None,
            "rir_audio": rir["audio"] if reverberant else None,
            "rir_channel": rir.get("channel", 0) if reverberant else None,
        }

    def __call__(self, speech, sample_id):
        recipe = self.recipe(sample_id)
        noise = self.loader(recipe["noise_audio"]) if recipe["noise_audio"] else None
        rir = (self.rir_loader(recipe["rir_audio"], recipe["rir_channel"])
               if recipe["rir_audio"] else None)
        output, measured = augment_waveform(
            speech,
            recipe["condition"],
            noise=noise,
            rir=rir,
            snr_db=recipe["snr_db"],
            noise_crop_start=recipe["noise_crop_start"] or 0,
            global_gain_db=recipe["global_gain_db"],
            tail_policy=self.settings["rir_tail_policy"],
            peak_ceiling=self.settings["peak_ceiling"],
        )
        return output, {**recipe, **measured}
