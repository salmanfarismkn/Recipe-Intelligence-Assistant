from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from recipe_intelligence.database import connect_database, ingest_dataset_archive
from recipe_intelligence.ingestion import DATASET_FILES
from recipe_intelligence.matching import (
    count_instruction_steps,
    match_recipes,
    parse_ingredient,
)


@pytest.fixture
def connection() -> sqlite3.Connection:
    database = connect_database(":memory:")
    yield database
    database.close()


def make_archive(path: Path) -> Path:
    records = {
        "garlic-chicken": {
            "title": "Garlic Chicken",
            "ingredients": [
                "1 1/2 cups skinless, boneless chicken breast halves ADVERTISEMENT",
                "2 cloves garlic, minced",
                "salt and ground black pepper to taste",
            ],
            "instructions": "1. Prepare chicken.\n2. Cook chicken.",
        },
        "peanut-garlic": {
            "title": "Peanut Garlic Spread",
            "ingredients": ["1 cup peanut butter", "2 cloves garlic"],
            "instructions": "Spread. Mix.",
        },
        "simple-garlic": {
            "title": "Simple Garlic",
            "ingredients": ["garlic"],
            "instructions": "Mix garlic.",
        },
        "too-many-steps": {
            "title": "Three Steps",
            "ingredients": ["garlic"],
            "instructions": "1. Prep.\n2. Cook.\n3. Serve.",
        },
        "ambiguous-steps": {
            "title": "Ambiguous Steps",
            "ingredients": ["garlic"],
            "instructions": "Mix garlic. Stir well.",
        },
        "empty-list": {"title": "Empty Ingredients", "ingredients": []},
        "missing-list": {"title": "Missing Ingredients"},
    }
    with zipfile.ZipFile(path, "w") as archive:
        for filename, source in DATASET_FILES.items():
            archive.writestr(
                filename,
                json.dumps(records if source == "ar" else {}),
            )
    return path


def test_ingredient_parser_handles_amounts_units_and_explicit_aliases() -> None:
    parsed = parse_ingredient(
        "1 1/2 cups skinless, boneless chicken breast halves ADVERTISEMENT"
    )
    assert parsed.quantity == "3/2"
    assert parsed.quantity_status == "parsed"
    assert parsed.unit == "cup"
    assert parsed.canonical_name == "chicken breast"
    assert parsed.original_text.endswith("ADVERTISEMENT")

    range_parsed = parse_ingredient("2 to 3 tablespoons sugar")
    assert range_parsed.quantity is None
    assert range_parsed.quantity_status == "unsupported_range"
    assert range_parsed.canonical_name == "sugar"
    assert parse_ingredient("2 cloves garlic, minced").canonical_name == "garlic"
    assert parse_ingredient("scallions").canonical_name == "green onion"
    assert parse_ingredient("eggs").canonical_name == "egg"


def test_instruction_step_count_marks_ambiguous_format_as_unknown() -> None:
    assert count_instruction_steps("1. Chop.\n2. Cook.").count == 2
    assert count_instruction_steps("Chop ingredients.\nCook until ready.").count == 2
    assert count_instruction_steps("Mix garlic. Stir well.").count is None
    assert count_instruction_steps("Mix garlic.").count == 1
    assert count_instruction_steps(None).count is None


def test_matching_ranks_explains_matches_and_enforces_hard_constraints(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    ingest_dataset_archive(connection, make_archive(tmp_path / "matching.zip"))

    report = match_recipes(
        connection,
        ("1 clove garlic", "chicken breasts", "eggs"),
        excluded_ingredients=("peanut",),
        max_instruction_steps=2,
    )
    results = {recipe.recipe_id: recipe for recipe in report.results}

    assert set(results) == {"garlic-chicken", "simple-garlic"}
    assert "peanut-garlic" not in results
    assert "too-many-steps" not in results
    assert "ambiguous-steps" not in results
    chicken = results["garlic-chicken"]
    assert chicken.score == pytest.approx(2 / 3)
    assert [match.canonical_name for match in chicken.matched_ingredients] == [
        "chicken breast",
        "garlic",
    ]
    assert chicken.missing_ingredients == (
        "salt and ground black pepper to taste",
    )
    assert chicken.additional_ingredients == ("eggs",)
    assert chicken.instruction_step_count == 2
    assert report.total_recipes_in_dataset == 7
    assert report.candidate_recipes_analyzed == 5
    assert report.rejected_excluded_ingredients == 1
    assert report.rejected_instruction_step_limit == 1
    assert report.rejected_unknown_instruction_steps == 1
    assert report.rejected_unknown_ingredient_records == 2
    assert report.rejected_unparseable_ingredient_records == 0
    assert not report.quantity_sufficiency_verified


def test_unparseable_hard_exclusion_inputs_and_invalid_limits_are_rejected(
    connection: sqlite3.Connection,
) -> None:
    with pytest.raises(ValueError, match="could not be parsed"):
        match_recipes(connection, ("!!!",))
    with pytest.raises(ValueError, match="maximum instruction steps"):
        match_recipes(connection, ("garlic",), max_instruction_steps=1.5)
    with pytest.raises(ValueError, match="at least one available"):
        match_recipes(connection, ())


def test_aliases_are_used_for_candidate_retrieval(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    records = {
        "scallion-soup": {
            "title": "Scallion Soup",
            "ingredients": ["1 bunch scallions"],
            "instructions": "Simmer.",
        }
    }
    with zipfile.ZipFile(tmp_path / "alias.zip", "w") as archive:
        for filename, source in DATASET_FILES.items():
            archive.writestr(
                filename,
                json.dumps(records if source == "ar" else {}),
            )
    ingest_dataset_archive(connection, tmp_path / "alias.zip")

    report = match_recipes(connection, ("green onion",))

    assert [recipe.recipe_id for recipe in report.results] == ["scallion-soup"]
