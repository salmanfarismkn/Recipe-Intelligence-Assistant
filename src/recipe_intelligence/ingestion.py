"""Dataset archive validation and inventory reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

DATASET_ARCHIVE_URL = "https://eightportions.com/recipes_raw.zip"
DATASET_ARCHIVE_SHA256 = "1ae20f3260313d501edf23c22c6f1875e87917f5a8e2ff13b40450719322c81b"
DATASET_FILES = {
    "recipes_raw_nosource_ar.json": "ar",
    "recipes_raw_nosource_epi.json": "epi",
    "recipes_raw_nosource_fn.json": "fn",
}
EXPECTED_RECORD_COUNTS = {"ar": 39802, "epi": 25323, "fn": 60039}
EXPECTED_FIELDS = ("title", "ingredients", "instructions", "picture_link")


class DatasetError(ValueError):
    """Raised when an archive cannot be safely interpreted as this dataset."""


class _DuplicateJSONKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"duplicate JSON object key: {key!r}")


@dataclass(frozen=True)
class RecipeRecord:
    source: str
    recipe_id: str
    original: dict[str, Any]


@dataclass(frozen=True)
class SourceInventory:
    source: str
    filename: str
    record_count: int
    expected_record_count: int
    record_count_matches_snapshot: bool
    empty_record_count: int
    missing_field_counts: dict[str, int]
    null_field_counts: dict[str, int]
    empty_ingredient_list_count: int
    blank_title_count: int
    blank_instruction_count: int
    invalid_record_count: int
    invalid_ingredient_list_count: int
    non_string_ingredient_count: int


@dataclass(frozen=True)
class DuplicateRecipeId:
    recipe_id: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class DatasetInventory:
    archive_sha256: str
    archive_matches_pinned_snapshot: bool
    total_record_count: int
    total_record_count_matches_snapshot: bool
    unique_recipe_id_count: int
    duplicate_recipe_ids_across_sources: tuple[DuplicateRecipeId, ...]
    sources: tuple[SourceInventory, ...]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKeyError(key)
        result[key] = value
    return result


def _archive_members(archive: zipfile.ZipFile) -> dict[str, str]:
    filenames = archive.namelist()
    duplicate_members = sorted(
        name for name, count in Counter(filenames).items() if count > 1
    )
    if duplicate_members:
        raise DatasetError(
            "archive contains duplicate member names: "
            + ", ".join(duplicate_members)
        )

    missing = sorted(set(DATASET_FILES) - set(filenames))
    if missing:
        raise DatasetError(
            "archive is missing required dataset files: " + ", ".join(missing)
        )

    unexpected_json = sorted(
        name
        for name in filenames
        if name.lower().endswith(".json") and name not in DATASET_FILES
    )
    if unexpected_json:
        raise DatasetError(
            "archive contains unexpected JSON files: " + ", ".join(unexpected_json)
        )

    return DATASET_FILES.copy()


def iter_recipe_records(archive_path: str | Path) -> Iterator[RecipeRecord]:
    """Yield every source record, preserving its source, ID, and original fields."""
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"cannot open dataset archive {archive_path}: {exc}") from exc

    with archive:
        source_files = _archive_members(archive)
        for filename, source in source_files.items():
            try:
                with archive.open(filename) as member:
                    records = json.load(member, object_pairs_hook=_unique_object)
            except (_DuplicateJSONKeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise DatasetError(f"invalid JSON in {filename}: {exc}") from exc

            if not isinstance(records, dict):
                raise DatasetError(
                    f"{filename} must contain a JSON object keyed by recipe ID"
                )

            for recipe_id, original in records.items():
                if not isinstance(original, dict):
                    raise DatasetError(
                        f"{filename} record {recipe_id!r} must be a JSON object"
                    )
                yield RecipeRecord(
                    source=source,
                    recipe_id=recipe_id,
                    original=original,
                )


def sha256_file(file_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_official_archive(destination: str | Path) -> Path:
    """Download the published archive and replace the destination only if verified."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        request = urllib.request.Request(
            DATASET_ARCHIVE_URL,
            headers={"User-Agent": "Recipe-Intelligence-Assistant/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination_path.parent,
                prefix=f"{destination_path.name}.",
                suffix=".download",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                digest = hashlib.sha256()
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    temporary_file.write(block)

        actual_hash = digest.hexdigest()
        if actual_hash != DATASET_ARCHIVE_SHA256:
            raise DatasetError(
                "downloaded archive checksum mismatch: "
                f"expected {DATASET_ARCHIVE_SHA256}, got {actual_hash}"
            )
        temporary_path.replace(destination_path)
        return destination_path
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def inspect_dataset_archive(archive_path: str | Path) -> DatasetInventory:
    """Validate all archive records and return counts without dropping any records."""
    source_counts: dict[str, dict[str, Any]] = {
        source: {
            "filename": filename,
            "record_count": 0,
            "empty_record_count": 0,
            "missing_field_counts": Counter(),
            "null_field_counts": Counter(),
            "empty_ingredient_list_count": 0,
            "blank_title_count": 0,
            "blank_instruction_count": 0,
            "invalid_record_count": 0,
            "invalid_ingredient_list_count": 0,
            "non_string_ingredient_count": 0,
        }
        for filename, source in DATASET_FILES.items()
    }
    id_sources: dict[str, set[str]] = {}
    total_records = 0

    for record in iter_recipe_records(archive_path):
        counts = source_counts[record.source]
        original = record.original
        total_records += 1
        counts["record_count"] += 1
        id_sources.setdefault(record.recipe_id, set()).add(record.source)
        if not original:
            counts["empty_record_count"] += 1

        for field in EXPECTED_FIELDS:
            if field not in original:
                counts["missing_field_counts"][field] += 1
            elif original[field] is None:
                counts["null_field_counts"][field] += 1

        if "title" in original and isinstance(original["title"], str):
            if not original["title"].strip():
                counts["blank_title_count"] += 1

        if "instructions" in original and isinstance(original["instructions"], str):
            if not original["instructions"].strip():
                counts["blank_instruction_count"] += 1

        if "ingredients" in original:
            ingredients = original["ingredients"]
            if not isinstance(ingredients, list):
                counts["invalid_ingredient_list_count"] += 1
            else:
                if not ingredients:
                    counts["empty_ingredient_list_count"] += 1
                counts["non_string_ingredient_count"] += sum(
                    not isinstance(item, str) for item in ingredients
                )

        invalid_record = any(
            field in original and original[field] is not None
            and not isinstance(original[field], expected_type)
            for field, expected_type in (
                ("title", str),
                ("ingredients", list),
                ("instructions", str),
                ("picture_link", str),
            )
        )
        if (
            isinstance(original.get("ingredients"), list)
            and any(not isinstance(item, str) for item in original["ingredients"])
        ):
            invalid_record = True
        if invalid_record:
            counts["invalid_record_count"] += 1

    duplicate_ids = tuple(
        DuplicateRecipeId(recipe_id, tuple(sorted(sources)))
        for recipe_id, sources in sorted(id_sources.items())
        if len(sources) > 1
    )
    source_reports = tuple(
        SourceInventory(
            source=source,
            filename=values["filename"],
            record_count=values["record_count"],
            expected_record_count=EXPECTED_RECORD_COUNTS[source],
            record_count_matches_snapshot=(
                values["record_count"] == EXPECTED_RECORD_COUNTS[source]
            ),
            empty_record_count=values["empty_record_count"],
            missing_field_counts=dict(values["missing_field_counts"]),
            null_field_counts=dict(values["null_field_counts"]),
            empty_ingredient_list_count=values["empty_ingredient_list_count"],
            blank_title_count=values["blank_title_count"],
            blank_instruction_count=values["blank_instruction_count"],
            invalid_record_count=values["invalid_record_count"],
            invalid_ingredient_list_count=values["invalid_ingredient_list_count"],
            non_string_ingredient_count=values["non_string_ingredient_count"],
        )
        for source, values in source_counts.items()
    )
    archive_sha256 = sha256_file(archive_path)
    return DatasetInventory(
        archive_sha256=archive_sha256,
        archive_matches_pinned_snapshot=archive_sha256 == DATASET_ARCHIVE_SHA256,
        total_record_count=total_records,
        total_record_count_matches_snapshot=(
            total_records == sum(EXPECTED_RECORD_COUNTS.values())
        ),
        unique_recipe_id_count=len(id_sources),
        duplicate_recipe_ids_across_sources=duplicate_ids,
        sources=source_reports,
    )


def _serialize(inventory: DatasetInventory) -> str:
    return json.dumps(asdict(inventory), indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and report the official Eight Portions recipe archive."
    )
    parser.add_argument("archive", type=Path, help="path to recipes_raw.zip")
    parser.add_argument(
        "--output",
        type=Path,
        help="write the JSON report to this path instead of standard output",
    )
    args = parser.parse_args(argv)

    try:
        report = inspect_dataset_archive(args.archive)
    except DatasetError as exc:
        parser.error(str(exc))

    serialized = _serialize(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + os.linesep, encoding="utf-8")
    else:
        sys.stdout.write(serialized + os.linesep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
