"""Download the pinned Eight Portions source archive."""

from __future__ import annotations

import argparse
from pathlib import Path

from recipe_intelligence.ingestion import download_official_archive


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the official Eight Portions dataset."
    )
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=Path("data/raw/recipes_raw.zip"),
    )
    destination = parser.parse_args().destination
    verified_path = download_official_archive(destination)
    print(f"Verified dataset archive: {verified_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
