"""Deterministic title, ingredient, and full-text recipe retrieval."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Literal

from recipe_intelligence.filters import RecipeFilters, build_filter_clause
from recipe_intelligence.normalization import normalize_ingredient

SearchField = Literal["title", "ingredients", "all"]
_RANKING_WEIGHTS = {
    "title": "0.0, 0.0, 5.0, 0.0, 0.0",
    "ingredients": "0.0, 0.0, 0.0, 5.0, 0.0",
    "all": "0.0, 0.0, 5.0, 2.0, 1.0",
}


@dataclass(frozen=True)
class RecipeSearchResult:
    source: str
    recipe_id: str
    title: str | None
    rank: float


@dataclass(frozen=True)
class RecipeDetail:
    source: str
    recipe_id: str
    original_record: dict[str, Any]
    validation_issues: tuple[str, ...]


def _fts_query(query: str, field: SearchField) -> str:
    if field not in {"title", "ingredients", "all"}:
        raise ValueError(f"unsupported search field: {field}")
    if not isinstance(query, str):
        raise TypeError("search query must be a string")

    tokens = normalize_ingredient(query).split()
    if not tokens:
        raise ValueError("search query must contain at least one searchable term")

    expression = " AND ".join(f'"{token}"' for token in tokens)
    if field == "all":
        return expression
    return f"{field} : ({expression})"


def search_recipes(
    connection: sqlite3.Connection,
    query: str,
    field: SearchField = "all",
    limit: int = 20,
    filters: RecipeFilters | None = None,
) -> list[RecipeSearchResult]:
    """Search indexed recipe text; lower BM25 ranks are more relevant."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")

    match_query = _fts_query(query, field)
    weights = _RANKING_WEIGHTS[field]
    filter_clause, filter_parameters = build_filter_clause(filters)
    rows = connection.execute(
        f"""
        SELECT
            recipes.source,
            recipes.recipe_id,
            recipes.title,
            bm25(recipe_fts, {weights}) AS rank
        FROM recipe_fts
        JOIN recipes
          ON recipes.source = recipe_fts.source
         AND recipes.recipe_id = recipe_fts.recipe_id
        WHERE recipe_fts MATCH ? AND {filter_clause}
        ORDER BY rank ASC, recipes.source ASC, recipes.recipe_id ASC
        LIMIT ?
        """,
        (match_query, *filter_parameters, limit),
    ).fetchall()
    return [
        RecipeSearchResult(
            source=row["source"],
            recipe_id=row["recipe_id"],
            title=row["title"],
            rank=row["rank"],
        )
        for row in rows
    ]


def search_title(
    connection: sqlite3.Connection,
    query: str,
    limit: int = 20,
    filters: RecipeFilters | None = None,
) -> list[RecipeSearchResult]:
    return search_recipes(
        connection, query, field="title", limit=limit, filters=filters
    )


def search_ingredients(
    connection: sqlite3.Connection,
    query: str,
    limit: int = 20,
    filters: RecipeFilters | None = None,
) -> list[RecipeSearchResult]:
    return search_recipes(
        connection, query, field="ingredients", limit=limit, filters=filters
    )


def search_natural_language(
    connection: sqlite3.Connection,
    query: str,
    limit: int = 20,
    filters: RecipeFilters | None = None,
) -> list[RecipeSearchResult]:
    return search_recipes(connection, query, field="all", limit=limit, filters=filters)


def get_recipe(
    connection: sqlite3.Connection, source: str, recipe_id: str
) -> RecipeDetail | None:
    row = connection.execute(
        """
        SELECT raw_record_json, validation_issues_json
        FROM recipes
        WHERE source = ? AND recipe_id = ?
        """,
        (source, recipe_id),
    ).fetchone()
    if row is None:
        return None

    return RecipeDetail(
        source=source,
        recipe_id=recipe_id,
        original_record=json.loads(row["raw_record_json"]),
        validation_issues=tuple(json.loads(row["validation_issues_json"])),
    )
