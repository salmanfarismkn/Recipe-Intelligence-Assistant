"""SQLite schema and atomic dataset persistence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from recipe_intelligence.ingestion import (
    EXPECTED_FIELDS,
    DatasetError,
    DatasetInventory,
    RecipeRecord,
    inspect_dataset_archive,
    iter_recipe_records,
    sha256_file,
)
from recipe_intelligence.normalization import normalize_ingredient
from recipe_intelligence.matching import count_instruction_steps, parse_ingredient


def connect_database(
    path: str | Path, *, check_same_thread: bool = True
) -> sqlite3.Connection:
    """Open and initialize a database connection with foreign keys enabled."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(path),
        check_same_thread=check_same_thread,
        timeout=5.0,
    )
    connection.row_factory = sqlite3.Row
    try:
        initialize_database(connection)
        connection.commit()
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.create_function(
        "normalize_ingredient", 1, normalize_ingredient, deterministic=True
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS recipes (
            source TEXT NOT NULL,
            recipe_id TEXT NOT NULL,
            title TEXT,
            instructions TEXT,
            picture_link TEXT,
            raw_record_json TEXT NOT NULL,
            validation_issues_json TEXT NOT NULL,
            ingredients_valid INTEGER NOT NULL DEFAULT 0,
            instruction_step_count INTEGER,
            instruction_step_method TEXT,
            PRIMARY KEY (source, recipe_id)
        );

        CREATE TABLE IF NOT EXISTS recipe_ingredients (
            source TEXT NOT NULL,
            recipe_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            original_text TEXT,
            original_value_json TEXT NOT NULL,
            normalized_text TEXT,
            canonical_name TEXT,
            quantity TEXT,
            quantity_status TEXT,
            unit TEXT,
            parse_warnings_json TEXT,
            PRIMARY KEY (source, recipe_id, position),
            FOREIGN KEY (source, recipe_id)
                REFERENCES recipes (source, recipe_id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS recipe_ingredients_normalized_idx
            ON recipe_ingredients (normalized_text);

        CREATE VIRTUAL TABLE IF NOT EXISTS recipe_fts USING fts5(
            source UNINDEXED,
            recipe_id UNINDEXED,
            title,
            ingredients,
            instructions
        );

        CREATE TABLE IF NOT EXISTS dataset_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversation_sessions (
            session_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY
        );
        """
    )
    recipe_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(recipes)")
    }
    if "ingredients_valid" not in recipe_columns:
        connection.execute(
            "ALTER TABLE recipes ADD COLUMN ingredients_valid INTEGER NOT NULL DEFAULT 0"
        )
    migration_complete = connection.execute(
        "SELECT 1 FROM schema_migrations WHERE version = 1"
    ).fetchone()
    if migration_complete is None:
        cursor = connection.execute(
            "SELECT source, recipe_id, raw_record_json FROM recipes"
        )
        while rows := cursor.fetchmany(500):
            updates = []
            for row in rows:
                original = json.loads(row["raw_record_json"])
                ingredients = original.get("ingredients")
                valid = isinstance(ingredients, list) and all(
                    isinstance(item, str) for item in ingredients
                )
                updates.append((int(valid), row["source"], row["recipe_id"]))
            connection.executemany(
                """
                UPDATE recipes SET ingredients_valid = ?
                WHERE source = ? AND recipe_id = ?
                """,
                updates,
            )
        connection.execute("INSERT INTO schema_migrations (version) VALUES (1)")

    recipe_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(recipes)")
    }
    if "instruction_step_count" not in recipe_columns:
        connection.execute(
            "ALTER TABLE recipes ADD COLUMN instruction_step_count INTEGER"
        )
    if "instruction_step_method" not in recipe_columns:
        connection.execute(
            "ALTER TABLE recipes ADD COLUMN instruction_step_method TEXT"
        )

    ingredient_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(recipe_ingredients)")
    }
    for column, column_type in (
        ("canonical_name", "TEXT"),
        ("quantity", "TEXT"),
        ("quantity_status", "TEXT"),
        ("unit", "TEXT"),
        ("parse_warnings_json", "TEXT"),
    ):
        if column not in ingredient_columns:
            connection.execute(
                f"ALTER TABLE recipe_ingredients ADD COLUMN {column} {column_type}"
            )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS recipe_ingredients_canonical_idx
        ON recipe_ingredients (canonical_name)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS recipe_ingredients_recipe_canonical_idx
        ON recipe_ingredients (source, recipe_id, canonical_name)
        """
    )

    migration_complete = connection.execute(
        "SELECT 1 FROM schema_migrations WHERE version = 2"
    ).fetchone()
    if migration_complete is None:
        cursor = connection.execute(
            "SELECT source, recipe_id, instructions FROM recipes"
        )
        while rows := cursor.fetchmany(500):
            updates = []
            for row in rows:
                analysis = count_instruction_steps(row["instructions"])
                updates.append(
                    (
                        analysis.count,
                        analysis.method,
                        row["source"],
                        row["recipe_id"],
                    )
                )
            connection.executemany(
                """
                UPDATE recipes
                SET instruction_step_count = ?, instruction_step_method = ?
                WHERE source = ? AND recipe_id = ?
                """,
                updates,
            )

        cursor = connection.execute(
            """
            SELECT source, recipe_id, position, original_text
            FROM recipe_ingredients
            """
        )
        while rows := cursor.fetchmany(500):
            updates = []
            for row in rows:
                parsed = (
                    parse_ingredient(row["original_text"])
                    if isinstance(row["original_text"], str)
                    else None
                )
                updates.append(
                    (
                        parsed.canonical_name if parsed else None,
                        parsed.quantity if parsed else None,
                        parsed.quantity_status if parsed else None,
                        parsed.unit if parsed else None,
                        (
                            json.dumps(parsed.warnings, ensure_ascii=False)
                            if parsed
                            else None
                        ),
                        row["source"],
                        row["recipe_id"],
                        row["position"],
                    )
                )
            connection.executemany(
                """
                UPDATE recipe_ingredients
                SET canonical_name = ?, quantity = ?, quantity_status = ?,
                    unit = ?, parse_warnings_json = ?
                WHERE source = ? AND recipe_id = ? AND position = ?
                """,
                updates,
            )
        connection.execute("INSERT INTO schema_migrations (version) VALUES (2)")


def _record_text(record: dict[str, Any], field: str) -> str | None:
    value = record.get(field)
    return value if isinstance(value, str) else None


def _validation_issues(record: RecipeRecord) -> list[str]:
    original = record.original
    expected_types = {
        "title": str,
        "ingredients": list,
        "instructions": str,
        "picture_link": str,
    }
    issues: list[str] = []

    for field in EXPECTED_FIELDS:
        if field not in original:
            issues.append(f"missing:{field}")
        elif original[field] is None:
            issues.append(f"null:{field}")
        elif not isinstance(original[field], expected_types[field]):
            issues.append(f"invalid_type:{field}")

    ingredients = original.get("ingredients")
    if isinstance(ingredients, list):
        if not ingredients:
            issues.append("empty:ingredients")
        for position, ingredient in enumerate(ingredients):
            if not isinstance(ingredient, str):
                issues.append(f"invalid_type:ingredients[{position}]")

    for field in ("title", "instructions"):
        value = original.get(field)
        if isinstance(value, str) and not value.strip():
            issues.append(f"blank:{field}")

    return issues


def _insert_record(connection: sqlite3.Connection, record: RecipeRecord) -> None:
    original = record.original
    instructions = _record_text(original, "instructions")
    step_analysis = count_instruction_steps(instructions)
    connection.execute(
        """
        INSERT INTO recipes (
            source, recipe_id, title, instructions, picture_link,
            raw_record_json, validation_issues_json, ingredients_valid,
            instruction_step_count, instruction_step_method
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.source,
            record.recipe_id,
            _record_text(original, "title"),
            _record_text(original, "instructions"),
            _record_text(original, "picture_link"),
            json.dumps(original, ensure_ascii=False, separators=(",", ":")),
            json.dumps(_validation_issues(record), ensure_ascii=False),
            int(
                isinstance(original.get("ingredients"), list)
                and all(
                    isinstance(item, str)
                    for item in original["ingredients"]
                )
            ),
            step_analysis.count,
            step_analysis.method,
        ),
    )

    ingredient_values: list[str] = []
    ingredients = original.get("ingredients")
    if isinstance(ingredients, list):
        for position, ingredient in enumerate(ingredients):
            original_text = ingredient if isinstance(ingredient, str) else None
            parsed = (
                parse_ingredient(original_text)
                if original_text is not None
                else None
            )
            normalized_text = (
                normalize_ingredient(original_text)
                if original_text is not None
                else None
            )
            connection.execute(
                """
                INSERT INTO recipe_ingredients (
                    source, recipe_id, position, original_text,
                    original_value_json, normalized_text, canonical_name,
                    quantity, quantity_status, unit, parse_warnings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.source,
                    record.recipe_id,
                    position,
                    original_text,
                    json.dumps(ingredient, ensure_ascii=False, separators=(",", ":")),
                    normalized_text,
                    parsed.canonical_name if parsed else None,
                    parsed.quantity if parsed else None,
                    parsed.quantity_status if parsed else None,
                    parsed.unit if parsed else None,
                    (
                        json.dumps(parsed.warnings, ensure_ascii=False)
                        if parsed
                        else None
                    ),
                ),
            )
            if normalized_text:
                ingredient_values.append(normalized_text)

    connection.execute(
        """
        INSERT INTO recipe_fts (source, recipe_id, title, ingredients, instructions)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            record.source,
            record.recipe_id,
            _record_text(original, "title") or "",
            " ".join(ingredient_values),
            instructions or "",
        ),
    )


def ingest_dataset_archive(
    connection: sqlite3.Connection, archive_path: str | Path
) -> DatasetInventory:
    """Atomically replace stored recipes with every record in the validated archive."""
    inventory = inspect_dataset_archive(archive_path)
    source_counts = {source.source: 0 for source in inventory.sources}
    inserted_count = 0

    with connection:
        connection.execute("DELETE FROM recipe_fts")
        connection.execute("DELETE FROM recipes")

        for record in iter_recipe_records(archive_path):
            _insert_record(connection, record)
            source_counts[record.source] += 1
            inserted_count += 1

        if inserted_count != inventory.total_record_count:
            raise DatasetError(
                "dataset record count changed during import: "
                f"inspected {inventory.total_record_count}, inserted {inserted_count}"
            )

        for source_inventory in inventory.sources:
            if source_counts[source_inventory.source] != source_inventory.record_count:
                raise DatasetError(
                    f"record count changed during import for {source_inventory.source}"
                )

        if sha256_file(archive_path) != inventory.archive_sha256:
            raise DatasetError("dataset archive changed while it was being imported")

        metadata = {
            "archive_sha256": inventory.archive_sha256,
            "total_record_count": str(inventory.total_record_count),
            "inventory_json": json.dumps(
                asdict(inventory), ensure_ascii=False, separators=(",", ":")
            ),
        }
        connection.executemany(
            """
            INSERT INTO dataset_metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            metadata.items(),
        )

    return inventory
