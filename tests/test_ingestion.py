from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from recipe_intelligence.ingestion import (
    DATASET_FILES,
    DatasetError,
    inspect_dataset_archive,
    iter_recipe_records,
)


def make_archive(
    path: Path,
    records_by_source: dict[str, list[tuple[str, dict[str, object]]]],
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for filename, source in DATASET_FILES.items():
            records = dict(records_by_source.get(source, []))
            archive.writestr(filename, json.dumps(records))
    return path


def test_inspection_counts_and_retains_source_scoped_records(tmp_path: Path) -> None:
    archive_path = make_archive(
        tmp_path / "test.zip",
        {
            "ar": [("shared-id", {"title": "test"})],
            "epi": [("shared-id", {})],
            "fn": [("fn-id", {"ingredients": [], "instructions": " "})],
        },
    )

    report = inspect_dataset_archive(archive_path)

    assert report.total_record_count == 3
    assert report.unique_recipe_id_count == 2
    assert not report.total_record_count_matches_snapshot
    assert report.duplicate_recipe_ids_across_sources[0].recipe_id == "shared-id"
    assert report.duplicate_recipe_ids_across_sources[0].sources == ("ar", "epi")
    sources = {source.source: source for source in report.sources}
    assert sources["epi"].empty_record_count == 1
    assert sources["epi"].missing_field_counts["title"] == 1
    assert not sources["epi"].record_count_matches_snapshot
    assert sources["fn"].empty_ingredient_list_count == 1
    assert sources["fn"].blank_instruction_count == 1
    assert [
        (record.source, record.recipe_id)
        for record in iter_recipe_records(archive_path)
    ] == [
        ("ar", "shared-id"),
        ("epi", "shared-id"),
        ("fn", "fn-id"),
    ]


def test_inspection_counts_null_and_invalid_values(tmp_path: Path) -> None:
    archive_path = make_archive(
        tmp_path / "test.zip",
        {
            "epi": [
                (
                    "test-id",
                    {
                        "title": None,
                        "ingredients": ["test ingredient", 1],
                        "instructions": None,
                        "picture_link": None,
                    },
                )
            ]
        },
    )

    epi = {
        source.source: source for source in inspect_dataset_archive(archive_path).sources
    }["epi"]

    assert epi.null_field_counts == {
        "title": 1,
        "instructions": 1,
        "picture_link": 1,
    }
    assert epi.invalid_record_count == 1
    assert epi.non_string_ingredient_count == 1


def test_duplicate_json_recipe_ids_are_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for filename in DATASET_FILES:
            if filename.endswith("_ar.json"):
                content = '{"duplicate-id": {}, "duplicate-id": {}}'
            else:
                content = "{}"
            archive.writestr(filename, content)

    with pytest.raises(DatasetError, match="duplicate JSON object key"):
        list(iter_recipe_records(archive_path))


def test_missing_source_file_is_reported(tmp_path: Path) -> None:
    archive_path = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("recipes_raw_nosource_ar.json", "{}")

    with pytest.raises(DatasetError, match="missing required dataset files"):
        inspect_dataset_archive(archive_path)


def test_unexpected_json_file_is_reported(tmp_path: Path) -> None:
    archive_path = make_archive(tmp_path / "unexpected.zip", {})
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr("other.json", "{}")

    with pytest.raises(DatasetError, match="unexpected JSON files"):
        inspect_dataset_archive(archive_path)
