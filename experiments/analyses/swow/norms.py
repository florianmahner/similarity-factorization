"""Load Glasgow Norms behavioral ratings.

Glasgow Norms (Scott et al., 2019) provide ratings for:
  arousal, valence, dominance, concreteness, imageability,
  familiarity, AoA, size, gender.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROPERTIES = [
    "arousal", "valence", "dominance", "concreteness",
    "imageability", "familiarity", "aoa", "size", "gender",
]


def load_behavioral_ratings(data_dir: str | Path) -> pd.DataFrame:
    """Load Glasgow Norms, returning word + 9 rating columns."""
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "semantic_norms/glasgow_norms.csv")
    df["word"] = df["word"].str.lower()
    df = df.rename(columns={"semsize": "size"})
    return df[["word"] + PROPERTIES]
