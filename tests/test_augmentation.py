import numpy as np
import pytest

from earnings_asr.augmentation import (
    OnlineAugmentationSampler,
    apply_rir,
    augment_waveform,
    mix_at_snr,
    prepare_rir,
)


def test_identity_rir_preserves_waveform_and_length():
    speech = np.random.default_rng(1).normal(0, 0.05, 16000).astype("float32")
    output, metadata = apply_rir(speech, np.array([1.0], dtype="float32"))
    assert np.allclose(output, speech, atol=1e-6)
    assert len(output) == len(speech)
    assert metadata["tail_policy"] == "crop"


def test_rir_leading_silence_trim_and_full_tail():
    rir = np.r_[np.zeros(12), 1.0, 0.5, 0.25].astype("float32")
    prepared, metadata = prepare_rir(rir)
    assert metadata["rir_trimmed_samples"] == 12
    assert np.linalg.norm(prepared) == pytest.approx(1.0)
    output, _ = apply_rir(np.ones(8, dtype="float32") * 0.01, rir, tail_policy="full")
    assert len(output) == 10


def test_known_active_speech_snr_and_short_noise_wrapping():
    speech = np.r_[np.zeros(1600), np.ones(6400) * 0.1, np.zeros(1600)].astype("float32")
    noise = np.random.default_rng(2).normal(0, 0.2, 613).astype("float32")
    output, metadata = mix_at_snr(speech, noise, 5.0, crop_start=407)
    assert metadata["actual_snr_db"] == pytest.approx(5.0, abs=1e-6)
    assert metadata["noise_wrapped"]
    assert len(output) == len(speech)
    assert np.isfinite(output).all()
    assert np.max(np.abs(output)) <= 0.990001


def test_combined_augmentation_is_deterministic_and_not_clipped():
    rng = np.random.default_rng(3)
    speech = rng.normal(0, 0.5, 4000).astype("float32")
    noise = rng.normal(0, 1.0, 721).astype("float32")
    rir = np.r_[np.zeros(20), 1.0, np.geomspace(0.4, 0.001, 500)].astype("float32")
    arguments = dict(noise=noise, rir=rir, snr_db=-5, noise_crop_start=211,
                     global_gain_db=3.0)
    first, first_meta = augment_waveform(speech, "noise_reverb", **arguments)
    second, second_meta = augment_waveform(speech, "noise_reverb", **arguments)
    assert np.array_equal(first, second)
    assert first_meta == second_meta
    assert len(first) == len(speech)
    assert np.max(np.abs(first)) <= 0.990001


def test_online_sampler_repeats_within_epoch_and_resamples_across_epochs():
    sampler = OnlineAugmentationSampler(
        {"clean": 0.25, "noise": 0.25, "reverb": 0.25, "noise_reverb": 0.25},
        [-5, 0, 5, 10, 15, 20],
        seed=42,
    )
    first = sampler.sample("utterance-1", 0, noise_frames=1000)
    assert first == sampler.sample("utterance-1", 0, noise_frames=1000)
    assert first != sampler.sample("utterance-1", 1, noise_frames=1000)
