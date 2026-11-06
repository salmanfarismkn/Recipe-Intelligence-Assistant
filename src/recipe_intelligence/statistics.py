"""Deterministic statistics computed over complete filtered recipe sets."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

from recipe_intelligence.filters import RecipeFilters, build_filter_clause
from recipe_intelligence.normalization import normalize_ingredient


@dataclass(frozen=True)
class StatisticResult:
    statistic: str
    value: int | float | None
    unit: str
    numerator: int
    denominator: int
    filters: dict[str, Any]
    analyzed_recipe_count: int
    excluded_recipe_count: int
    calculation_method: str


def _filters_payload(filters: RecipeFilters | None) -> dict[str, Any]:
    return asdict(filters if filters is not None else RecipeFilters())


def _filtered_recipes(
    connection: sqlite3.Connection, filters: RecipeFilters | None
) -> list[sqlite3.Row]:
    where, parameters = build_filter_clause(filters)
    return connection.execute(
        f"""
        SELECT source, recipe_id, raw_record_json
        FROM recipes
        WHERE {where}
        ORDER BY source, recipe_id
        """,
        parameters,
    ).fetchall()


def _valid_ingredient_lists(
    rows: list[sqlite3.Row],
) -> list[tuple[dict[str, Any], list[str]]]:
    valid: list[tuple[dict[str, Any], list[str]]] = []
    for row in rows:
        original = json.loads(row["raw_record_json"])
        ingredients = original.get("ingredients")
        if isinstance(ingredients, list) and all(
            isinstance(item, str) for item in ingredients
        ):
            valid.append((original, ingredients))
    return valid


def calculate_recipe_count(
    connection: sqlite3.Connection,
    filters: RecipeFilters | None = None,
) -> StatisticResult:
    rows = _filtered_recipes(connection, filters)
    count = len(rows)
    return StatisticResult(
        statistic="recipe_count",
        value=count,
        unit="recipes",
        numerator=count,
        denominator=count,
        filters=_filters_payload(filters),
        analyzed_recipe_count=count,
        excluded_recipe_count=0,
        calculation_method="Count every recipe record matching the supplied filters.",
    )


def calculate_average_ingredient_lines(
    connection: sqlite3.Connection,
    filters: RecipeFilters | None = None,
) -> StatisticResult:
    rows = _filtered_recipes(connection, filters)
    valid = _valid_ingredient_lists(rows)
    line_count = sum(len(ingredients) for _, ingredients in valid)
    denominator = len(valid)
    return StatisticResult(
        statistic="average_ingredient_lines",
        value=line_count / denominator if denominator else None,
        unit="ingredient lines per recipe",
        numerator=line_count,
        denominator=denominator,
        filters=_filters_payload(filters),
        analyzed_recipe_count=len(rows),
        excluded_recipe_count=len(rows) - denominator,
        calculation_method=(
            "Sum ingredient-list entries and divide by recipes with a present "
            "list containing only strings; empty lists count as zero."
        ),
    )


def calculate_ingredient_prevalence(
    connection: sqlite3.Connection,
    ingredient: str,
    filters: RecipeFilters | None = None,
) -> StatisticResult:
    query_tokens = set(normalize_ingredient(ingredient).split())
    if not query_tokens:
        raise ValueError("ingredient query must contain searchable text")

    rows = _filtered_recipes(connection, filters)
    valid = _valid_ingredient_lists(rows)
    matching_recipe_count = 0
    for _, ingredients in valid:
        recipe_tokens = {
            token
            for line in ingredients
            for token in normalize_ingredient(line).split()
        }
        if query_tokens.issubset(recipe_tokens):
            matching_recipe_count += 1

    denominator = len(valid)
    return StatisticResult(
        statistic="ingredient_prevalence",
        value=matching_recipe_count / denominator if denominator else None,
        unit="fraction of recipes",
        numerator=matching_recipe_count,
        denominator=denominator,
        filters=_filters_payload(filters),
        analyzed_recipe_count=len(rows),
        excluded_recipe_count=len(rows) - denominator,
        calculation_method=(
            "Count each recipe once when all normalized query tokens occur "
            "in its ingredient lines, then divide by recipes with valid "
            "ingredient lists. No aliases or stemming are applied."
        ),
    )
