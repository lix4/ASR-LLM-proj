import numpy as np
import pytest
import soundfile as sf

from earnings_asr.evaluation import _quality_summary, render_manifest_row
from earnings_asr.scoring import score_pair


def test_fixed_corruption_row_renders_reproducibly(tmp_path):
    rng = np.random.default_rng(7)
    speech = rng.normal(0, 0.05, 3200).astype("float32")
    noise = rng.normal(0, 0.1, 701).astype("float32")
    rir = np.column_stack((np.r_[1.0, np.zeros(31)], np.r_[0.0, 1.0, np.zeros(30)]))
    speech_path, noise_path, rir_path = (tmp_path / name for name in
                                         ("speech.wav", "noise.wav", "rir.wav"))
    sf.write(speech_path, speech, 16000, subtype="FLOAT")
    sf.write(noise_path, noise, 16000, subtype="FLOAT")
    sf.write(rir_path, rir, 16000, subtype="FLOAT")
    row = {
        "id": "example:noise_reverb:5dB", "condition": "noise_reverb",
        "speech_audio": str(speech_path), "noise_audio": str(noise_path),
        "rir_audio": str(rir_path), "rir_channel": 1, "snr_db": 5.0,
        "noise_crop_start": 123, "global_gain_db": -1.0,
    }
    settings = {"rir_tail_policy": "crop", "peak_ceiling": 0.99}

    first, first_metadata = render_manifest_row(row, settings)
    second, second_metadata = render_manifest_row(row, settings)

    assert np.array_equal(first, second)
    assert first_metadata == second_metadata
    assert len(first) == len(speech)
    assert first_metadata["actual_snr_db"] == pytest.approx(5.0, abs=1e-6)


def test_quality_summary_reports_condition_and_snr_corpus_wer():
    rows = []
    for condition, snr, reference, prediction in (
        ("clean", None, "one two", "one two"),
        ("noise", -5.0, "one two", "one"),
        ("noise", 5.0, "one two", "one two"),
        ("reverb", None, "one two", "one wrong"),
        ("noise_reverb", -5.0, "one two", ""),
    ):
        rows.append({"condition": condition, "snr_db": snr, "noise_category": "noise",
                     **score_pair(reference, prediction)})

    summary = _quality_summary(rows)

    assert summary["by_condition"]["clean"]["wer"] == 0.0
    assert summary["by_condition"]["noise"]["wer"] == pytest.approx(0.25)
    assert summary["by_condition_and_snr_db"]["noise"]["-5.0"]["wer"] == 0.5
    assert summary["robust_wer_macro"] == pytest.approx((0.25 + 0.5 + 1.0) / 3)
