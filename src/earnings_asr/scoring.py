"""Versioned, symmetric normalization and corpus-level edit counts."""

import re
import unicodedata

from jiwer import process_words

NORMALIZATION = "english_v1_nfkc_casefold_punctuation_space_preserve_numbers"


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    # Keep apostrophes within words, decimal points within numbers, and numeric commas.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    chars = []
    for i, char in enumerate(text):
        before = text[i - 1] if i else ""
        after = text[i + 1] if i + 1 < len(text) else ""
        if char == "'" and before.isalnum() and after.isalnum():
            chars.append(char)
        elif char == "." and before.isdigit() and after.isdigit():
            chars.append(char)
        elif unicodedata.category(char).startswith(("P", "S")):
            chars.append(" ")
        else:
            chars.append(char)
    return " ".join("".join(chars).split())


def score_pair(reference, prediction):
    reference, prediction = normalize(reference), normalize(prediction)
    result = process_words(reference, prediction)
    return {
        "reference_normalized": reference,
        "prediction_normalized": prediction,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "reference_words": result.hits + result.substitutions + result.deletions,
    }


def aggregate(rows):
    counts = {key: sum(row[key] for row in rows) for key in
              ("substitutions", "deletions", "insertions", "reference_words")}
    errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    return {**counts, "wer": errors / counts["reference_words"] if counts["reference_words"] else None,
            "normalization": NORMALIZATION, "samples": len(rows)}
