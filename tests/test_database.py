from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from recipe_intelligence.database import connect_database, ingest_dataset_archive
from recipe_intelligence.ingestion import DATASET_FILES, DatasetError
from recipe_intelligence.normalization import normalize_ingredient
from recipe_intelligence.retrieval import (
    get_recipe,
    search_ingredients,
    search_natural_language,
    search_title,
)


def make_archive(
    path: Path,
    records_by_source: dict[str, dict[str, dict[str, object]]],
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for filename, source in DATASET_FILES.items():
            archive.writestr(
                filename,
                json.dumps(records_by_source.get(source, {}), ensure_ascii=False),
            )
    return path


@pytest.fixture
def connection() -> sqlite3.Connection:
    database = connect_database(":memory:")
    yield database
    database.close()


def test_normalization_preserves_words_and_quantity_tokens() -> None:
    assert normalize_ingredient("  1/2 tsp. Crème fraîche — finely chopped ") == (
        "1 2 tsp crème fraîche finely chopped"
    )


def test_ingestion_preserves_original_values_and_normalizes_separately(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    archive_path = make_archive(
        tmp_path / "recipes.zip",
        {
            "ar": {
                "stable-id": {
                    "title": "Original Title ",
                    "ingredients": [" 1 clove GARLIC ", "Crème fraîche"],
                    "instructions": "Mix exactly as written.\n",
                    "picture_link": None,
                    "unrecognized_source_field": {"kept": True},
                }
            },
            "epi": {"empty-id": {}},
        },
    )

    report = ingest_dataset_archive(connection, archive_path)

    stored = connection.execute(
        "SELECT source, recipe_id, raw_record_json FROM recipes ORDER BY recipe_id"
    ).fetchall()
    assert [(row["source"], row["recipe_id"]) for row in stored] == [
        ("epi", "empty-id"),
        ("ar", "stable-id"),
    ]
    detail = get_recipe(connection, "ar", "stable-id")
    assert detail is not None
    assert detail.original_record["title"] == "Original Title "
    assert detail.original_record["ingredients"] == [
        " 1 clove GARLIC ",
        "Crème fraîche",
    ]
    assert detail.original_record["instructions"] == "Mix exactly as written.\n"
    assert detail.original_record["unrecognized_source_field"] == {"kept": True}
    assert "null:picture_link" in detail.validation_issues
    ingredient_rows = connection.execute(
        """
        SELECT position, original_text, normalized_text, canonical_name,
               quantity, quantity_status, unit
        FROM recipe_ingredients
        WHERE source = 'ar' AND recipe_id = 'stable-id'
        ORDER BY position
        """
    ).fetchall()
    assert [(row["original_text"], row["normalized_text"]) for row in ingredient_rows] == [
        (" 1 clove GARLIC ", "1 clove garlic"),
        ("Crème fraîche", "crème fraîche"),
    ]
    assert (
        ingredient_rows[0]["canonical_name"],
        ingredient_rows[0]["quantity"],
        ingredient_rows[0]["quantity_status"],
        ingredient_rows[0]["unit"],
    ) == ("garlic", "1", "parsed", "clove")
    step_metadata = connection.execute(
        """
        SELECT instruction_step_count, instruction_step_method
        FROM recipes WHERE source = 'ar' AND recipe_id = 'stable-id'
        """
    ).fetchone()
    assert tuple(step_metadata) == (1, "single_instruction_block")
    assert report.total_record_count == 2
    assert connection.execute(
        "SELECT value FROM dataset_metadata WHERE key = 'total_record_count'"
    ).fetchone()["value"] == "2"


def test_ingestion_is_repeatable_and_replaces_the_prior_snapshot(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    first_archive = make_archive(
        tmp_path / "first.zip",
        {"ar": {"first-id": {"title": "First", "ingredients": []}}},
    )
    second_archive = make_archive(
        tmp_path / "second.zip",
        {"fn": {"second-id": {"title": "Second", "ingredients": []}}},
    )

    ingest_dataset_archive(connection, first_archive)
    ingest_dataset_archive(connection, first_archive)
    assert connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0] == 1

    ingest_dataset_archive(connection, second_archive)
    assert connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0] == 1
    assert get_recipe(connection, "ar", "first-id") is None
    assert get_recipe(connection, "fn", "second-id") is not None


def test_invalid_import_keeps_the_existing_snapshot(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    valid_archive = make_archive(
        tmp_path / "valid.zip",
        {"ar": {"existing-id": {"title": "Existing"}}},
    )
    invalid_archive = tmp_path / "duplicate-id.zip"
    with zipfile.ZipFile(invalid_archive, "w") as archive:
        for filename in DATASET_FILES:
            content = (
                '{"duplicate-id": {}, "duplicate-id": {}}'
                if filename.endswith("_ar.json")
                else "{}"
            )
            archive.writestr(filename, content)

    ingest_dataset_archive(connection, valid_archive)
    with pytest.raises(DatasetError, match="duplicate JSON object key"):
        ingest_dataset_archive(connection, invalid_archive)

    assert get_recipe(connection, "ar", "existing-id") is not None
    assert connection.execute("SELECT COUNT(*) FROM recipes").fetchone()[0] == 1


def test_title_ingredient_and_natural_language_search(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    archive_path = make_archive(
        tmp_path / "search.zip",
        {
            "ar": {
                "chicken-id": {
                    "title": "Roasted Chicken Soup",
                    "ingredients": ["1 clove garlic", "2 cups water"],
                    "instructions": "Bake until tender.",
                },
                "vegetable-id": {
                    "title": "Vegetable Stew",
                    "ingredients": ["onion", "potato"],
                    "instructions": "Simmer until soft.",
                },
            }
        },
    )
    ingest_dataset_archive(connection, archive_path)

    assert [result.recipe_id for result in search_title(connection, "roasted chicken")] == [
        "chicken-id"
    ]
    assert [result.recipe_id for result in search_ingredients(connection, "garlic")] == [
        "chicken-id"
    ]
    assert [
        result.recipe_id
        for result in search_natural_language(connection, "garlic bake")
    ] == ["chicken-id"]
    assert get_recipe(connection, "missing", "unknown") is None


def test_search_rejects_empty_terms_and_invalid_limits(
    connection: sqlite3.Connection,
) -> None:
    with pytest.raises(ValueError, match="searchable term"):
        search_natural_language(connection, "!?")
    with pytest.raises(ValueError, match="limit"):
        search_title(connection, "chicken", limit=0)


def test_legacy_database_migration_persists_ingredient_validity(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    legacy = sqlite3.connect(database_path)
    legacy.execute(
        """
        CREATE TABLE recipes (
            source TEXT NOT NULL,
            recipe_id TEXT NOT NULL,
            title TEXT,
            instructions TEXT,
            picture_link TEXT,
            raw_record_json TEXT NOT NULL,
            validation_issues_json TEXT NOT NULL,
            PRIMARY KEY (source, recipe_id)
        )
        """
    )
    legacy.execute(
        """
        CREATE TABLE recipe_ingredients (
            source TEXT NOT NULL,
            recipe_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            original_text TEXT,
            original_value_json TEXT NOT NULL,
            normalized_text TEXT,
            PRIMARY KEY (source, recipe_id, position),
            FOREIGN KEY (source, recipe_id)
                REFERENCES recipes (source, recipe_id)
                ON DELETE CASCADE
        )
        """
    )
    legacy.executemany(
        """
        INSERT INTO recipes (
            source, recipe_id, instructions, raw_record_json,
            validation_issues_json
        ) VALUES ('ar', ?, ?, ?, '[]')
        """,
        [
            (
                "valid",
                "1. Prepare.\n2. Cook.",
                json.dumps({"ingredients": ["1 clove garlic"]}),
            ),
            ("missing", None, json.dumps({"title": "No ingredient field"})),
            ("invalid", None, json.dumps({"ingredients": ["garlic", 1]})),
        ],
    )
    legacy.execute(
        """
        INSERT INTO recipe_ingredients (
            source, recipe_id, position, original_text,
            original_value_json, normalized_text
        ) VALUES ('ar', 'valid', 0, '1 clove garlic', '"1 clove garlic"',
                  '1 clove garlic')
        """
    )
    legacy.commit()
    legacy.close()

    migrated = connect_database(database_path)
    values = {
        row["recipe_id"]: row["ingredients_valid"]
        for row in migrated.execute(
            "SELECT recipe_id, ingredients_valid FROM recipes"
        )
    }
    migrated_metadata = migrated.execute(
        """
        SELECT instruction_step_count, instruction_step_method
        FROM recipes WHERE source = 'ar' AND recipe_id = 'valid'
        """
    ).fetchone()
    migrated_ingredient = migrated.execute(
        """
        SELECT canonical_name, quantity, quantity_status, unit
        FROM recipe_ingredients
        WHERE source = 'ar' AND recipe_id = 'valid'
        """
    ).fetchone()
    migrated.close()

    reopened = connect_database(database_path)
    persisted_values = {
        row["recipe_id"]: row["ingredients_valid"]
        for row in reopened.execute(
            "SELECT recipe_id, ingredients_valid FROM recipes"
        )
    }
    reopened.close()

    assert values == {"valid": 1, "missing": 0, "invalid": 0}
    assert persisted_values == values
    assert tuple(migrated_metadata) == (2, "numbered_steps")
    assert tuple(migrated_ingredient) == ("garlic", "1", "parsed", "clove")
