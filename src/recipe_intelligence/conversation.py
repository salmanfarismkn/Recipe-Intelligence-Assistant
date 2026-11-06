"""Structured, session-isolated conversation state."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, replace
from typing import Sequence

from recipe_intelligence.filters import RecipeFilters
from recipe_intelligence.normalization import normalize_ingredient


@dataclass(frozen=True)
class RecipeRef:
    source: str
    recipe_id: str


@dataclass(frozen=True)
class ConversationState:
    session_id: str
    active_recipe: RecipeRef | None = None
    filters: RecipeFilters = RecipeFilters()
    available_ingredients: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    previous_result_ids: tuple[RecipeRef, ...] = ()
    current_task: str | None = None
    result_limit: int = 20
    last_statistic_ingredient: str | None = None
    max_instruction_steps: int | None = None
    ranking_preferences: tuple[str, ...] = ()
    search_query: str | None = None
    search_field: str | None = None
    details_requested: bool = False
    dietary_restrictions: tuple[str, ...] = ()
    hard_exclusions: tuple[str, ...] = ()
    substitution_ingredient: str | None = None

    @property
    def excluded_ingredients(self) -> tuple[str, ...]:
        return self.filters.exclude_ingredients


def new_session_state(session_id: str) -> ConversationState:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session ID must be a non-empty string")
    return ConversationState(session_id=session_id)


def _merge_ingredient_terms(
    existing: tuple[str, ...], additions: Sequence[str]
) -> tuple[str, ...]:
    merged = list(existing)
    seen = {normalize_ingredient(value) for value in existing}
    for value in additions:
        normalized = normalize_ingredient(value)
        if not normalized:
            raise ValueError("ingredient restrictions must contain searchable text")
        if normalized not in seen:
            merged.append(value)
            seen.add(normalized)
    return tuple(merged)


def apply_restrictions(
    state: ConversationState,
    *,
    include_ingredients: Sequence[str] = (),
    exclude_ingredients: Sequence[str] = (),
    title_query: str | None = None,
    sources: tuple[str, ...] | None = None,
) -> ConversationState:
    """Merge new restrictions and make recipe search the active task."""
    updated_filters = replace(
        state.filters,
        sources=state.filters.sources if sources is None else sources,
        title_query=(
            state.filters.title_query if title_query is None else title_query
        ),
        include_ingredients=_merge_ingredient_terms(
            state.filters.include_ingredients, include_ingredients
        ),
        exclude_ingredients=_merge_ingredient_terms(
            state.filters.exclude_ingredients, exclude_ingredients
        ),
    )
    return replace(state, filters=updated_filters, current_task="recipe_search")


def replace_filters(
    state: ConversationState, filters: RecipeFilters
) -> ConversationState:
    return replace(state, filters=filters, current_task="recipe_search")


def update_user_context(
    state: ConversationState,
    *,
    available_ingredients: Sequence[str] | None = None,
    preferences: Sequence[str] | None = None,
) -> ConversationState:
    available = state.available_ingredients
    if available_ingredients is not None:
        available = _merge_ingredient_terms((), available_ingredients)

    updated_preferences = state.preferences
    if preferences is not None:
        updated_preferences = tuple(preferences)

    return replace(
        state,
        available_ingredients=available,
        preferences=updated_preferences,
    )


def set_active_recipe(
    state: ConversationState, source: str, recipe_id: str
) -> ConversationState:
    if not source or not recipe_id:
        raise ValueError("active recipe requires both source and recipe ID")
    return replace(
        state,
        active_recipe=RecipeRef(source=source, recipe_id=recipe_id),
        current_task="recipe_detail",
    )


def record_search_results(
    state: ConversationState, results: Sequence[RecipeRef]
) -> ConversationState:
    return replace(
        state,
        previous_result_ids=tuple(results),
        current_task="recipe_search",
    )


def _state_payload(state: ConversationState) -> dict[str, object]:
    return {
        "active_recipe": (
            asdict(state.active_recipe) if state.active_recipe is not None else None
        ),
        "filters": {
            "sources": list(state.filters.sources),
            "title_query": state.filters.title_query,
            "include_ingredients": list(state.filters.include_ingredients),
            "exclude_ingredients": list(state.filters.exclude_ingredients),
        },
        "available_ingredients": list(state.available_ingredients),
        "preferences": list(state.preferences),
        "previous_result_ids": [
            asdict(recipe) for recipe in state.previous_result_ids
        ],
        "current_task": state.current_task,
        "result_limit": state.result_limit,
        "last_statistic_ingredient": state.last_statistic_ingredient,
        "max_instruction_steps": state.max_instruction_steps,
        "ranking_preferences": list(state.ranking_preferences),
        "search_query": state.search_query,
        "search_field": state.search_field,
        "details_requested": state.details_requested,
        "dietary_restrictions": list(state.dietary_restrictions),
        "hard_exclusions": list(state.hard_exclusions),
        "substitution_ingredient": state.substitution_ingredient,
    }


def save_session_state(
    connection: sqlite3.Connection, state: ConversationState
) -> None:
    if not state.session_id.strip():
        raise ValueError("session ID must be a non-empty string")
    with connection:
        connection.execute(
            """
            INSERT INTO conversation_sessions (session_id, state_json)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                state.session_id,
                json.dumps(_state_payload(state), ensure_ascii=False),
            ),
        )


def load_session_state(
    connection: sqlite3.Connection, session_id: str
) -> ConversationState:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session ID must be a non-empty string")

    row = connection.execute(
        "SELECT state_json FROM conversation_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return new_session_state(session_id)

    payload = json.loads(row["state_json"])
    active = payload["active_recipe"]
    filters = payload["filters"]
    return ConversationState(
        session_id=session_id,
        active_recipe=RecipeRef(**active) if active is not None else None,
        filters=RecipeFilters(
            sources=tuple(filters["sources"]),
            title_query=filters["title_query"],
            include_ingredients=tuple(filters["include_ingredients"]),
            exclude_ingredients=tuple(filters["exclude_ingredients"]),
        ),
        available_ingredients=tuple(payload["available_ingredients"]),
        preferences=tuple(payload["preferences"]),
        previous_result_ids=tuple(
            RecipeRef(**recipe) for recipe in payload["previous_result_ids"]
        ),
        current_task=payload["current_task"],
        result_limit=payload.get("result_limit", 20),
        last_statistic_ingredient=payload.get("last_statistic_ingredient"),
        max_instruction_steps=payload.get("max_instruction_steps"),
        ranking_preferences=tuple(payload.get("ranking_preferences", ())),
        search_query=payload.get("search_query"),
        search_field=payload.get("search_field"),
        details_requested=payload.get("details_requested", False),
        dietary_restrictions=tuple(payload.get("dietary_restrictions", ())),
        hard_exclusions=tuple(payload.get("hard_exclusions", ())),
        substitution_ingredient=payload.get("substitution_ingredient"),
    )
