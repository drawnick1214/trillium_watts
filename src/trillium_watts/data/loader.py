"""Data loading — download from Google Drive and read CSV files."""

from __future__ import annotations

from pathlib import Path

import gdown
import pandas as pd


def download_from_google_drive(file_id: str, output_path: str | Path) -> Path:
    """Download a file from Google Drive by its file ID using gdown."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(output_path), quiet=False)
    return output_path


def load_raw_data(
    path: str | Path,
    separator: str = ";",
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """Load a raw CSV file with the specified separator and encoding."""
    return pd.read_csv(path, sep=separator, encoding=encoding)
