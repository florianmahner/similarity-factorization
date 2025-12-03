from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_swow_data(data_dir: Path, use_all_responses: bool) -> pd.DataFrame:
    """Load SWOW strength data (cue, response, count, strength)."""
    suffix = "R123" if use_all_responses else "R1"
    df = pd.read_csv(data_dir / f"strength.SWOW-EN.{suffix}.20180827.csv", sep="\t")
    df = pd.DataFrame(
        {
            "cue": df["cue"],
            "response": df["response"],
            "count": df[suffix],
            "strength": df[f"{suffix}.Strength"],
        }
    )
    return df.dropna(subset=["response"])


def filter_by_word_length(
    cues: list[str], responses: list[str], counts: list[float], min_length: int
) -> tuple[list[str], list[str], list[float]]:
    """Keep only edges where both words meet minimum length."""
    filtered = [
        (c, r, cnt) for c, r, cnt in zip(cues, responses, counts)
        if isinstance(c, str) and isinstance(r, str) and
        len(c) >= min_length and len(r) >= min_length
    ]
    if not filtered:
        return [], [], []
    return zip(*filtered) if filtered else ([], [], [])


def filter_to_top_words(
    cues: list[str], responses: list[str], counts: list[float], top_n: int
) -> tuple[list[str], list[str], list[float]]:
    """Keep only edges where both words are in the top N most frequent."""
    all_words = pd.Series(cues + responses)
    top_words = set(all_words.value_counts().head(top_n).index)
    filtered = [
        (c, r, cnt) for c, r, cnt in zip(cues, responses, counts)
        if c in top_words and r in top_words
    ]
    if not filtered:
        return [], [], []
    return zip(*filtered) if filtered else ([], [], [])
