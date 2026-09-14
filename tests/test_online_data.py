import numpy as np
import pytest

from earnings_asr.online_data import OnlineWaveformAugmenter


@pytest.fixture
def settings():
    return {
        "condition_probabilities": {
            "clean": 0.0, "noise": 0.0, "reverb": 0.0, "noise_reverb": 1.0,
        },
        "snr_db": [0, 10],
        "global_gain_db": [-1, 1],
        "rir_tail_policy": "crop",
        "peak_ceiling": 0.99,
    }


def test_online_augmenter_uses_only_train_resources_and_changes_epoch(settings):
    resources = {
        "noise.wav": np.random.default_rng(1).normal(0, 0.1, 800).astype("float32"),
        "rir.wav": np.array([1.0, 0.2, 0.1], dtype="float32"),
    }
    loader = resources.__getitem__
    augmenter = OnlineWaveformAugmenter(
        settings,
        [{"id": "noise", "audio": "noise.wav", "frames": 800, "split": "train"}],
        [{"id": "rir", "audio": "rir.wav", "split": "train"}],
        loader=loader,
        rir_loader=lambda path, channel: resources[path],
    )
    speech = np.ones(1600, dtype="float32") * 0.05
    first_audio, first = augmenter(speech, "sample")
    repeat_audio, repeat = augmenter(speech, "sample")
    assert np.array_equal(first_audio, repeat_audio)
    assert first == repeat
    augmenter.set_epoch(1)
    second_audio, second = augmenter(speech, "sample")
    assert first["seed"] != second["seed"]
    assert not np.array_equal(first_audio, second_audio)
    assert first["noise_id"] == "noise" and first["rir_id"] == "rir"


def test_online_augmenter_rejects_held_out_resources(settings):
    with pytest.raises(ValueError, match="train"):
        OnlineWaveformAugmenter(
            settings,
            [{"id": "noise", "audio": "noise.wav", "frames": 10, "split": "test"}],
            [{"id": "rir", "audio": "rir.wav", "split": "train"}],
        )
