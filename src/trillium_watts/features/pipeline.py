"""Feature engineering pipeline — orchestrates temporal + cyclic encoding."""

from __future__ import annotations

import pandas as pd

from trillium_watts.features.cyclic import encode_cyclic_features
from trillium_watts.features.temporal import extract_temporal_features


def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run temporal extraction and cyclic encoding on the DataFrame."""
    df = extract_temporal_features(df)
    df = encode_cyclic_features(df)
    return df
