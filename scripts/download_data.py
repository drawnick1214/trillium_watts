"""Download raw data from Google Drive."""

import sys
from pathlib import Path

# Add src to path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trillium_watts.config import load_config, get_project_root
from trillium_watts.data.loader import download_from_google_drive


def main():
    config = load_config()
    root = get_project_root()
    output_path = root / config.data.raw_data_path

    print(f"Downloading data to {output_path}...")
    download_from_google_drive(config.data.google_drive_file_id, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
