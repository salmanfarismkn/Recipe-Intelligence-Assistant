from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from recipe_intelligence.conversation import (
    RecipeRef,
    apply_restrictions,
    load_session_state,
    new_session_state,
    record_search_results,
    save_session_state,
    set_active_recipe,
    update_user_context,
)
from recipe_intelligence.database import connect_database, ingest_dataset_archive
from recipe_intelligence.filters import RecipeFilters
from recipe_intelligence.ingestion import DATASET_FILES
from recipe_intelligence.retrieval import search_ingredients
from recipe_intelligence.statistics import (
    calculate_average_ingredient_lines,
    calculate_ingredient_prevalence,
    calculate_recipe_count,
)


@pytest.fixture
def connection() -> sqlite3.Connection:
    database = connect_database(":memory:")
    yield database
    database.close()


def make_archive(path: Path) -> Path:
    records = {
        "garlic-soup": {
            "title": "Garlic Soup",
            "ingredients": ["garlic", "onion"],
            "instructions": "Simmer.",
        },
        "garlic-bread": {
            "title": "Garlic Bread",
            "ingredients": ["garlic", "butter"],
            "instructions": "Bake.",
        },
        "onion-stew": {
            "title": "Onion Stew",
            "ingredients": ["onion", "potato"],
            "instructions": "Cook.",
        },
        "empty-list": {
            "title": "Empty List",
            "ingredients": [],
            "instructions": "Serve.",
        },
        "missing-list": {"title": "Missing List"},
    }
    with zipfile.ZipFile(path, "w") as archive:
        for filename, source in DATASET_FILES.items():
            archive.writestr(
                filename,
                json.dumps(records if source == "ar" else {}),
            )
    return path


@pytest.fixture
def imported_database(
    connection: sqlite3.Connection, tmp_path: Path
) -> sqlite3.Connection:
    ingest_dataset_archive(connection, make_archive(tmp_path / "recipes.zip"))
    return connection


def test_filters_apply_before_search_limit_and_exclude_unknown_ingredients(
    imported_database: sqlite3.Connection,
) -> None:
    filters = RecipeFilters(
        include_ingredients=("garlic",),
        exclude_ingredients=("onion",),
    )

    results = search_ingredients(
        imported_database, "garlic", limit=1, filters=filters
    )

    assert [result.recipe_id for result in results] == ["garlic-bread"]


def test_statistics_use_complete_filtered_dataset_and_report_denominators(
    imported_database: sqlite3.Connection,
) -> None:
    filters = RecipeFilters(sources=("ar",))

    count = calculate_recipe_count(imported_database, filters)
    average = calculate_average_ingredient_lines(imported_database, filters)
    prevalence = calculate_ingredient_prevalence(
        imported_database, "garlic", filters
    )

    assert (count.value, count.denominator, count.analyzed_recipe_count) == (5, 5, 5)
    assert count.filters["sources"] == ("ar",)
    assert (average.numerator, average.denominator) == (6, 4)
    assert average.value == 1.5
    assert average.analyzed_recipe_count == 5
    assert average.excluded_recipe_count == 1
    assert (prevalence.numerator, prevalence.denominator) == (2, 4)
    assert prevalence.value == 0.5
    assert prevalence.analyzed_recipe_count == 5
    assert prevalence.calculation_method


def test_statistics_apply_title_and_exclusion_filters(
    imported_database: sqlite3.Connection,
) -> None:
    filters = RecipeFilters(
        title_query="Garlic",
        exclude_ingredients=("onion",),
    )

    result = calculate_recipe_count(imported_database, filters)

    assert result.value == 1
    assert result.denominator == 1
    assert result.filters["exclude_ingredients"] == ("onion",)


def test_conversation_state_persists_without_cross_session_contamination(
    imported_database: sqlite3.Connection,
) -> None:
    state = new_session_state("session-a")
    state = update_user_context(
        state,
        available_ingredients=("garlic", "butter"),
        preferences=("vegetarian",),
    )
    state = set_active_recipe(state, "ar", "garlic-bread")
    state = record_search_results(state, (RecipeRef("ar", "garlic-bread"),))
    state = apply_restrictions(state, exclude_ingredients=("onion",))
    state = apply_restrictions(
        state,
        exclude_ingredients=("ONION", "dairy"),
        include_ingredients=("garlic",),
    )
    save_session_state(imported_database, state)

    restored = load_session_state(imported_database, "session-a")
    unrelated = load_session_state(imported_database, "session-b")

    assert restored == state
    assert restored.excluded_ingredients == ("onion", "dairy")
    assert restored.current_task == "recipe_search"
    assert restored.active_recipe == RecipeRef("ar", "garlic-bread")
    assert restored.previous_result_ids == (RecipeRef("ar", "garlic-bread"),)
    assert unrelated.filters == RecipeFilters()
    assert unrelated.available_ingredients == ()
    assert unrelated.previous_result_ids == ()
    assert unrelated.current_task is None


def test_blank_ingredient_statistics_query_is_rejected(
    imported_database: sqlite3.Connection,
) -> None:
    with pytest.raises(ValueError, match="searchable text"):
        calculate_ingredient_prevalence(imported_database, "!?")
