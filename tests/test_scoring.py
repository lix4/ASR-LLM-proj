import pytest

from earnings_asr.scoring import aggregate, normalize, score_pair


def test_corpus_wer_uses_total_word_denominator():
    rows = [score_pair("one", "wrong"), score_pair("one two three four five six seven eight nine", "one two three four five six seven eight nine")]
    assert aggregate(rows)["wer"] == pytest.approx(0.1)


def test_empty_prediction_and_silent_reference():
    result = aggregate([score_pair("one two", ""), score_pair("", "hallucination")])
    assert result["deletions"] == 2
    assert result["insertions"] == 1
    assert result["wer"] == 1.5
    assert aggregate([score_pair("", "")])["wer"] is None


def test_financial_normalization_is_explicit_and_symmetric():
    text = "ACME’s revenue: $1,234.50, up 5%! Twenty-one."
    assert normalize(text) == "acme's revenue 1234.50 up 5 twenty one"
    assert score_pair(text, text)["substitutions"] == 0
    assert normalize("twenty") != normalize("20")
