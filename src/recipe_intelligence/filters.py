"""Reusable, explainable recipe filters."""

from __future__ import annotations

from dataclasses import dataclass

from recipe_intelligence.normalization import normalize_ingredient


@dataclass(frozen=True)
class RecipeFilters:
    sources: tuple[str, ...] = ()
    title_query: str | None = None
    include_ingredients: tuple[str, ...] = ()
    exclude_ingredients: tuple[str, ...] = ()


def _ingredient_filter(value: str) -> str:
    normalized = normalize_ingredient(value)
    if not normalized:
        raise ValueError("ingredient filters must contain searchable text")
    return normalized


def build_filter_clause(filters: RecipeFilters | None) -> tuple[str, list[str]]:
    """Build parameterized SQL for filters shared by search and statistics."""
    if filters is None:
        return "1", []

    conditions: list[str] = []
    parameters: list[str] = []

    if filters.sources:
        if any(not source for source in filters.sources):
            raise ValueError("source filters cannot contain empty values")
        placeholders = ", ".join("?" for _ in filters.sources)
        conditions.append(f"recipes.source IN ({placeholders})")
        parameters.extend(filters.sources)

    if filters.title_query is not None:
        conditions.append(
            "instr(normalize_ingredient(COALESCE(recipes.title, '')), ?) > 0"
        )
        title_query = normalize_ingredient(filters.title_query)
        if not title_query:
            raise ValueError("title filter must contain searchable text")
        parameters.append(title_query)

    if filters.exclude_ingredients:
        conditions.append("recipes.ingredients_valid = 1")

    for ingredient in filters.include_ingredients:
        conditions.append(
            """
            EXISTS (
                SELECT 1
                FROM recipe_ingredients AS included_ingredient
                WHERE included_ingredient.source = recipes.source
                  AND included_ingredient.recipe_id = recipes.recipe_id
                  AND instr(
                      ' ' || included_ingredient.normalized_text || ' ',
                      ' ' || ? || ' '
                  ) > 0
            )
            """
        )
        parameters.append(_ingredient_filter(ingredient))

    for ingredient in filters.exclude_ingredients:
        conditions.append(
            """
            NOT EXISTS (
                SELECT 1
                FROM recipe_ingredients AS excluded_ingredient
                WHERE excluded_ingredient.source = recipes.source
                  AND excluded_ingredient.recipe_id = recipes.recipe_id
                  AND instr(
                      ' ' || excluded_ingredient.normalized_text || ' ',
                      ' ' || ? || ' '
                  ) > 0
            )
            """
        )
        parameters.append(_ingredient_filter(ingredient))

    return (" AND ".join(conditions) if conditions else "1"), parameters
