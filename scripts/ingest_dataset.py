"""Import the validated dataset archive into SQLite."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from recipe_intelligence.database import connect_database, ingest_dataset_archive


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import all source records into the local SQLite recipe store."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/recipes.sqlite3"),
    )
    args = parser.parse_args()

    connection = connect_database(args.database)
    try:
        inventory = ingest_dataset_archive(connection, args.archive)
    finally:
        connection.close()

    print(json.dumps(asdict(inventory), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
