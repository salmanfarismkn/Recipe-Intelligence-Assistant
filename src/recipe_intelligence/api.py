"""FastAPI routes and persisted, context-aware conversational sessions."""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Generator, Literal, Sequence

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from fastapi.staticfiles import StaticFiles

from recipe_intelligence.assistant import (
    InterpretedRequest,
    execute_request,
    generate_grounded_response,
    interpret_request,
)
from recipe_intelligence.conversation import (
    ConversationState,
    RecipeRef,
    load_session_state,
    new_session_state,
    save_session_state,
)
from recipe_intelligence.database import connect_database
from recipe_intelligence.filters import RecipeFilters
from recipe_intelligence.llm import (
    JSONProvider,
    LLMError,
    provider_from_environment,
)
from recipe_intelligence.matching import (
    IngredientMatchReport,
    match_recipes,
    normalize_ranking_preferences,
)
from recipe_intelligence.normalization import normalize_ingredient
from recipe_intelligence.retrieval import get_recipe, search_recipes
from recipe_intelligence.statistics import (
    calculate_average_ingredient_lines,
    calculate_ingredient_prevalence,
    calculate_recipe_count,
)
from recipe_intelligence.substitution import (
    SubstitutionProposal,
    propose_substitutions,
)

logger = logging.getLogger("recipe_intelligence.api")
_SESSION_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_SESSION_MESSAGE_MAX = 2_000
_REQUEST_BODY_MAX = 64 * 1024
_DAIRY_SEARCH_EXCLUSIONS = (
    "butter",
    "buttermilk",
    "casein",
    "cheese",
    "condensed milk",
    "cream",
    "cream cheese",
    "ghee",
    "half and half",
    "ice cream",
    "lactose",
    "milk",
    "milk powder",
    "milk protein",
    "milk solids",
    "nonfat dry milk",
    "skim milk",
    "sour cream",
    "whey",
    "yogurt",
)


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FilterModel(APIModel):
    sources: list[Literal["ar", "epi", "fn"]] = Field(default_factory=list, max_length=3)
    title_query: str | None = Field(default=None, max_length=300)
    include_ingredients: list[str] = Field(default_factory=list, max_length=30)
    exclude_ingredients: list[str] = Field(default_factory=list, max_length=30)


class RecipeSearchRequest(APIModel):
    query: str | None = Field(default=None, min_length=1, max_length=300)
    natural_language: str | None = Field(default=None, min_length=1, max_length=2_000)
    field: Literal["title", "ingredients", "all"] = "all"
    filters: FilterModel = Field(default_factory=FilterModel)
    limit: int = Field(default=20, ge=1, le=100)
    include_details: bool = False


class AnalyticsRequest(APIModel):
    statistic: Literal[
        "recipe_count", "average_ingredient_lines", "ingredient_prevalence"
    ]
    ingredient: str | None = Field(default=None, max_length=300)
    filters: FilterModel = Field(default_factory=FilterModel)


class MatchingRequest(APIModel):
    available_ingredients: list[str] = Field(min_length=1, max_length=100)
    excluded_ingredients: list[str] = Field(default_factory=list, max_length=100)
    max_instruction_steps: int | None = Field(default=None, ge=1, le=100)
    ranking_preferences: list[
        Literal["fewer_missing", "fewer_additional", "fewer_steps"]
    ] = Field(default_factory=list, max_length=3)
    limit: int = Field(default=20, ge=1, le=100)


class SubstitutionRequest(APIModel):
    recipe_id: str = Field(min_length=1, max_length=300)
    source: Literal["ar", "epi", "fn"] | None = None
    ingredient_to_replace: str = Field(min_length=1, max_length=300)
    replacement_ingredients: list[str] = Field(default_factory=list, max_length=20)
    excluded_ingredients: list[str] = Field(default_factory=list, max_length=100)
    dietary_restrictions: list[str] = Field(default_factory=list, max_length=30)
    preferences: list[str] = Field(default_factory=list, max_length=30)
    limit: int = Field(default=10, ge=1, le=20)


class SessionMessageRequest(APIModel):
    message: str = Field(min_length=1, max_length=_SESSION_MESSAGE_MAX)


class RequestSizeLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                response = _error(
                    "request_too_large",
                    f"Request body must not exceed {self.max_bytes} bytes.",
                    413,
                )
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)


def _filters(model: FilterModel) -> RecipeFilters:
    return RecipeFilters(
        sources=tuple(model.sources),
        title_query=model.title_query,
        include_ingredients=tuple(model.include_ingredients),
        exclude_ingredients=tuple(model.exclude_ingredients),
    )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _error(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _session_lock(app: FastAPI, session_id: str) -> threading.RLock:
    guard: threading.Lock = app.state.session_lock_guard
    with guard:
        lock = app.state.session_locks.get(session_id)
        if lock is None:
            lock = threading.RLock()
            app.state.session_locks[session_id] = lock
            app.state.session_lock_users[session_id] = 0
        app.state.session_lock_users[session_id] += 1
        return lock


@contextmanager
def _session_operation(app: FastAPI, session_id: str) -> Generator[None, None, None]:
    lock = _session_lock(app, session_id)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        guard: threading.Lock = app.state.session_lock_guard
        with guard:
            remaining = app.state.session_lock_users[session_id] - 1
            if remaining == 0:
                del app.state.session_lock_users[session_id]
                del app.state.session_locks[session_id]
            else:
                app.state.session_lock_users[session_id] = remaining


def _validate_session_id(session_id: str) -> None:
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_session_id",
                "message": "Session ID must be a UUID.",
            },
        )


def _recipe_reference_for_id(
    connection: sqlite3.Connection, recipe_id: str, source: str | None = None
) -> RecipeRef | None:
    if source is not None:
        return (
            RecipeRef(source, recipe_id)
            if get_recipe(connection, source, recipe_id) is not None
            else None
        )
    rows = connection.execute(
        "SELECT source FROM recipes WHERE recipe_id = ? ORDER BY source LIMIT 2",
        (recipe_id,),
    ).fetchall()
    if len(rows) > 1:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ambiguous_recipe_id",
                "message": "Recipe ID exists in multiple sources; specify source.",
            },
        )
    return RecipeRef(rows[0]["source"], recipe_id) if rows else None


def _request_defaults(task: str, **values: Any) -> InterpretedRequest:
    defaults: dict[str, Any] = {
        "task": task,
        "search_query": None,
        "dish_name": None,
        "search_field": None,
        "include_ingredients": (),
        "exclude_ingredients": (),
        "sources": (),
        "details_requested": False,
        "statistic": None,
        "statistic_ingredient": None,
        "available_ingredients": (),
        "max_instruction_steps": None,
        "ranking_preferences": (),
        "recipe_reference_text": None,
        "ingredient_to_replace": None,
        "replacement_ingredient": None,
        "dietary_restrictions": (),
        "preferences": (),
        "limit": 20,
    }
    defaults.update(values)
    return InterpretedRequest(**defaults)


def _reference_text(reference: RecipeRef) -> str:
    return f"{reference.source}:{reference.recipe_id}"


def _merge_terms(existing: Sequence[str], additions: Sequence[str]) -> tuple[str, ...]:
    merged = list(existing)
    seen = {normalize_ingredient(term) for term in merged}
    for term in additions:
        normalized = normalize_ingredient(term)
        if normalized and normalized not in seen:
            merged.append(term)
            seen.add(normalized)
    return tuple(merged)


def _lookup_reference(
    connection: sqlite3.Connection,
    state: ConversationState,
    message: str,
) -> tuple[RecipeRef | None, str | None]:
    ordinal = re.search(r"\b(first|1st|second|2nd|third|3rd)\b", message, re.I)
    if ordinal:
        index = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}[
            ordinal.group(1).casefold()
        ]
        if index >= len(state.previous_result_ids):
            return None, "That result is not in the previous search results."
        reference = state.previous_result_ids[index]
        if get_recipe(connection, reference.source, reference.recipe_id) is None:
            return None, "That previous result is no longer available."
        return reference, None
    if state.active_recipe is not None and get_recipe(
        connection, state.active_recipe.source, state.active_recipe.recipe_id
    ):
        return state.active_recipe, None
    if len(state.previous_result_ids) == 1:
        reference = state.previous_result_ids[0]
        if get_recipe(connection, reference.source, reference.recipe_id):
            return reference, None
    if len(state.previous_result_ids) > 1:
        return None, "Which result do you mean? Please say first, second, or third."
    return None, "Which recipe should I use? Search for a recipe first or provide its ID."


def _parse_fallback_request(
    message: str, state: ConversationState
) -> InterpretedRequest | None:
    normalized = message.strip()
    lowered = normalized.casefold()
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def parsed_number(value: str) -> int | None:
        return number_words.get(value.casefold(), int(value) if value.isdigit() else None)

    match = re.match(r"^(?:find|search for|search|show me)\s+(.+)$", normalized, re.I)
    if match:
        query = match.group(1).strip().rstrip(".?!")
        query = re.sub(
            r"^(?:a|an|the)\s+(?:recipe|recipes)\s+for\s+",
            "",
            query,
            flags=re.IGNORECASE,
        )
        query = re.sub(r"^(?:a|an|the)\s+", "", query, flags=re.IGNORECASE)
        query = re.sub(
            r"\s+(?:recipe|recipes)$", "", query, flags=re.IGNORECASE
        )
        query = re.sub(
            r"^(?:recipe|recipes)\s+for\s+", "", query, flags=re.IGNORECASE
        )
        query = re.sub(
            r"\s+(?:and\s+)?(?:show|include)\s+(?:me\s+)?(?:the\s+)?"
            r"(?:details|instructions)\b.*$",
            "",
            query,
            flags=re.IGNORECASE,
        )
        exclusions_match = re.search(
            r"\b(?:without|exclude|avoid)\s+(.+?)(?=\s+\b(?:show|include)\b|$)",
            query,
            flags=re.IGNORECASE,
        )
        exclusions = (
            tuple(
                item.strip(" .!?")
                for item in re.split(
                    r",|\band\b", exclusions_match.group(1), flags=re.IGNORECASE
                )
                if item.strip(" .!?")
            )
            if exclusions_match
            else ()
        )
        query = re.sub(
            r"\s+(?:without|exclude|avoid)\s+.+$",
            "",
            query,
            flags=re.IGNORECASE,
        )
        search_context = state.current_task == "recipe_search"
        return _request_defaults(
            "recipe_search",
            search_query=query,
            dish_name=None,
            search_field="all",
            exclude_ingredients=_merge_terms(
                state.filters.exclude_ingredients if search_context else (),
                exclusions,
            ),
            include_ingredients=(
                state.filters.include_ingredients if search_context else ()
            ),
            sources=state.filters.sources if search_context else (),
            details_requested="details" in lowered or "instructions" in lowered,
        )
    step_ranking_match = re.search(
        r"\b(?:among|show|find)\s+(?:the\s+)?recipes?\s+"
        r"(?:containing|with)\s+(.+?),?\s+(?:show\s+)?"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:recipes?\s+)?with\s+(?:the\s+)?fewest\s+"
        r"(?:instruction\s+)?steps?\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if step_ranking_match:
        ingredient = step_ranking_match.group(1).strip(" ,.!?")
        result_limit = parsed_number(step_ranking_match.group(2)) or 5
        return _request_defaults(
            "ingredient_matching",
            available_ingredients=(ingredient,),
            ranking_preferences=("fewer_steps",),
            limit=result_limit,
        )
    combined_prevalence = re.search(
        r"\bhow many recipes?\s+(?:contain|include|have)\s+"
        r"(?:both\s+)?(.+?)[?.!]*$",
        normalized,
        flags=re.IGNORECASE,
    )
    if combined_prevalence:
        ingredient_terms = [
            item.strip(" .!?")
            for item in re.split(
                r",|\band\b", combined_prevalence.group(1), flags=re.IGNORECASE
            )
            if item.strip(" .!?")
        ]
        if ingredient_terms:
            return _request_defaults(
                "recipe_search",
                statistic="ingredient_prevalence",
                statistic_ingredient=" ".join(ingredient_terms),
            )
    if re.search(r"\bhow many recipes\b", lowered):
        return _request_defaults("recipe_search", statistic="recipe_count")
    if re.search(r"\baverage\b.*\bingredient", lowered):
        return _request_defaults(
            "recipe_search", statistic="average_ingredient_lines"
        )
    prevalence_match = re.search(
        r"\b(?:how common is|prevalence of|what fraction of recipes contain)\s+"
        r"(.+?)[?.!]*$",
        normalized,
        flags=re.IGNORECASE,
    )
    if prevalence_match:
        ingredient = prevalence_match.group(1).strip(" .?!")
        return _request_defaults(
            "recipe_search",
            statistic="ingredient_prevalence",
            statistic_ingredient=ingredient,
        )
    if re.search(r"\bi have\b", lowered):
        ingredient_text = re.sub(
            r"^.*?\bi have\b", "", normalized, flags=re.IGNORECASE
        )
        ingredient_text = re.split(
            r";|\bbut\b|\bavoid\b|\bwithout\b|\bexclude\b|"
            r"\bno more than\b|\bat most\b|\bmaximum\b|\bmax\b|"
            r"\bsuggest\b|\bfind\b|\bgive me\b|\bprefer\b",
            ingredient_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        available = tuple(
            part.strip(" ,.!?")
            for part in re.split(r",|\band\b", ingredient_text, flags=re.I)
            if part.strip(" ,.!?")
        )
        if available:
            exclusion_match = re.search(
                r"\b(?:avoid|without|exclude)\s+(.+?)(?=;|\b(?:no more than|"
                r"at most|maximum|max|suggest|find|give me|prefer)\b|"
                r"\band\s+(?:with\s+)?(?:no more than|at most|maximum|max)\b|$)",
                normalized,
                flags=re.IGNORECASE,
            )
            exclusions = (
                tuple(
                    item.strip(" .!?")
                    for item in re.split(
                        r",|\band\b",
                        exclusion_match.group(1),
                        flags=re.IGNORECASE,
                    )
                    if item.strip(" .!?")
                )
                if exclusion_match
                else state.hard_exclusions
            )
            step_match = re.search(
                r"\b(?:no more than|at most|maximum|max)\s+"
                r"(one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
                r"\s+(?:instruction\s+)?steps?\b",
                normalized,
                flags=re.IGNORECASE,
            )
            result_match = re.search(
                r"\b(?:suggest|show|give me)\s+"
                r"(one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
                r"\s+(?:recipes|results)\b",
                normalized,
                flags=re.IGNORECASE,
            )
            return _request_defaults(
                "ingredient_matching",
                available_ingredients=available,
                exclude_ingredients=exclusions,
                max_instruction_steps=(
                    parsed_number(step_match.group(1))
                    if step_match
                    else state.max_instruction_steps
                ),
                ranking_preferences=state.ranking_preferences,
                limit=(
                    parsed_number(result_match.group(1))
                    if result_match
                    else 20
                ),
            )
    return None


def create_app(
    database_path: str | Path | None = None,
    *,
    provider: JSONProvider | None = None,
    use_environment_provider: bool = True,
) -> FastAPI:
    """Create the API application; pass a provider or use environment settings."""
    app = FastAPI(
        title="Recipe Intelligence Assistant",
        version="0.7.0",
        description="Deterministic recipe intelligence with evidence-grounded conversations.",
    )
    app.state.database_path = str(
        database_path
        or os.environ.get(
            "RECIPE_DATABASE_PATH", "data/processed/recipes.sqlite3"
        )
    )
    app.state.provider = provider
    if provider is None and use_environment_provider:
        app.state.provider = provider_from_environment()
    app.state.session_locks = {}
    app.state.session_lock_users = {}
    app.state.session_lock_guard = threading.Lock()

    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=_REQUEST_BODY_MAX,
    )

    def database_connection() -> Generator[sqlite3.Connection, None, None]:
        connection = connect_database(
            app.state.database_path,
            check_same_thread=False,
        )
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            connection.close()

    from fastapi import Depends

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        _ = request, error
        return _error("invalid_request", "Request validation failed.", 422)

    @app.exception_handler(HTTPException)
    async def http_error_handler(
        request: Request, error: HTTPException
    ) -> JSONResponse:
        _ = request
        detail = error.detail
        if isinstance(detail, dict) and "code" in detail:
            return _error(detail["code"], detail["message"], error.status_code)
        return _error("http_error", str(detail), error.status_code)

    @app.exception_handler(sqlite3.Error)
    async def database_error_handler(
        request: Request, error: sqlite3.Error
    ) -> JSONResponse:
        logger.exception("Database request failed: %s", request.url.path)
        _ = error
        return _error("database_error", "Database operation failed.", 503)

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request, error: ValueError
    ) -> JSONResponse:
        logger.info("Invalid operation for %s", request.url.path)
        _ = error
        return _error("invalid_operation", str(error), 422)

    @app.get("/health")
    def health(
        connection: sqlite3.Connection = Depends(database_connection),
    ) -> dict[str, str]:
        connection.execute("SELECT 1").fetchone()
        return {
            "status": "ok",
            "database": "ok",
            "gemini": "configured" if app.state.provider is not None else "disabled",
        }

    @app.post("/api/v1/recipes/search")
    def recipe_search(
        body: RecipeSearchRequest,
        connection: sqlite3.Connection = Depends(database_connection),
    ) -> dict[str, Any]:
        filters = _filters(body.filters)
        query = body.query
        field = body.field
        include_details = body.include_details
        if body.natural_language:
            if app.state.provider is None:
                interpreted = _parse_fallback_request(
                    body.natural_language, new_session_state("api-search")
                )
                if interpreted is None or interpreted.task != "recipe_search":
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "code": "llm_unavailable",
                            "message": "Natural-language interpretation is unavailable.",
                        },
                    )
            else:
                try:
                    resolution = interpret_request(
                        body.natural_language, app.state.provider
                    )
                except LLMError as error:
                    logger.warning(
                        "Gemini interpretation unavailable (%s)",
                        type(error).__name__,
                    )
                    interpreted = _parse_fallback_request(
                        body.natural_language, new_session_state("api-search")
                    )
                    if interpreted is None or interpreted.task != "recipe_search":
                        raise HTTPException(
                            status_code=503,
                            detail={
                                "code": "llm_unavailable",
                                "message": "Natural-language interpretation failed.",
                            },
                        ) from error
                else:
                    interpreted = resolution.request
                if interpreted is None or interpreted.task != "recipe_search":
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "code": "invalid_search_intent",
                            "message": "The request did not resolve to a recipe search.",
                        },
                    )
            query = interpreted.search_query or interpreted.dish_name
            field = interpreted.search_field or "all"
            include_details = (
                body.include_details or interpreted.details_requested
            )
            filters = RecipeFilters(
                sources=tuple(body.filters.sources) or interpreted.sources,
                title_query=body.filters.title_query,
                include_ingredients=(
                    tuple(body.filters.include_ingredients)
                    or interpreted.include_ingredients
                ),
                exclude_ingredients=(
                    tuple(body.filters.exclude_ingredients)
                    or interpreted.exclude_ingredients
                ),
            )
        if query is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_query",
                    "message": "Provide query or natural_language.",
                },
            )
        results = search_recipes(
            connection, query, field=field, limit=body.limit, filters=filters
        )
        items = []
        for result in results:
            item: dict[str, Any] = {
                "source": result.source,
                "recipe_id": result.recipe_id,
                "reference": _reference_text(
                    RecipeRef(result.source, result.recipe_id)
                ),
                "title": result.title,
                "rank": result.rank,
            }
            if include_details:
                detail = get_recipe(connection, result.source, result.recipe_id)
                if detail is not None:
                    item["ingredients"] = detail.original_record.get("ingredients")
                    item["instructions"] = detail.original_record.get("instructions")
            items.append(item)
        return {
            "query": query,
            "field": field,
            "filters_applied": _jsonable(filters),
            "result_count": len(items),
            "results": items,
        }

    @app.get("/api/v1/recipes/{recipe_id:path}")
    def recipe_details(
        recipe_id: str,
        source: Literal["ar", "epi", "fn"] | None = Query(default=None),
        connection: sqlite3.Connection = Depends(database_connection),
    ) -> dict[str, Any]:
        reference = _recipe_reference_for_id(connection, recipe_id, source)
        if reference is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "recipe_not_found", "message": "Recipe not found."},
            )
        detail = get_recipe(connection, reference.source, reference.recipe_id)
        assert detail is not None
        return {
            "source": detail.source,
            "recipe_id": detail.recipe_id,
            "reference": _reference_text(reference),
            "recipe": detail.original_record,
            "validation_issues": list(detail.validation_issues),
        }

    @app.post("/api/v1/analytics")
    def analytics(
        body: AnalyticsRequest,
        connection: sqlite3.Connection = Depends(database_connection),
    ) -> dict[str, Any]:
        filters = _filters(body.filters)
        if body.statistic == "recipe_count":
            result = calculate_recipe_count(connection, filters)
        elif body.statistic == "average_ingredient_lines":
            result = calculate_average_ingredient_lines(connection, filters)
        else:
            if not body.ingredient:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "missing_ingredient",
                        "message": "ingredient_prevalence requires ingredient.",
                    },
                )
            result = calculate_ingredient_prevalence(
                connection, body.ingredient, filters
            )
        return {"result": _jsonable(result)}

    @app.post("/api/v1/matching")
    def matching(
        body: MatchingRequest,
        connection: sqlite3.Connection = Depends(database_connection),
    ) -> dict[str, Any]:
        report = match_recipes(
            connection,
            body.available_ingredients,
            excluded_ingredients=body.excluded_ingredients,
            max_instruction_steps=body.max_instruction_steps,
            ranking_preferences=body.ranking_preferences,
            limit=body.limit,
        )
        serialized = _jsonable(report)
        serialized["score_explanation"] = (
            "Each recipe score is matched ingredient lines divided by parsed "
            "required ingredient lines. Hard constraints are applied before ranking."
        )
        serialized["hard_constraints"] = {
            "excluded_ingredients": list(report.excluded_ingredients),
            "maximum_instruction_steps": report.max_instruction_steps,
            "eligibility_changed_by_preferences": False,
        }
        return {"report": serialized}

    @app.post("/api/v1/substitutions")
    def substitutions(
        body: SubstitutionRequest,
        connection: sqlite3.Connection = Depends(database_connection),
    ) -> dict[str, Any]:
        reference = _recipe_reference_for_id(connection, body.recipe_id, body.source)
        if reference is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "recipe_not_found", "message": "Recipe not found."},
            )
        proposal = propose_substitutions(
            connection,
            reference.source,
            reference.recipe_id,
            body.ingredient_to_replace,
            replacement_ingredients=body.replacement_ingredients,
            excluded_ingredients=body.excluded_ingredients,
            dietary_restrictions=body.dietary_restrictions,
            preferences=body.preferences,
            limit=body.limit,
        )
        return {"proposal": _jsonable(proposal)}

    @app.post("/api/v1/sessions", status_code=201)
    def create_session(
        connection: sqlite3.Connection = Depends(database_connection),
    ) -> dict[str, Any]:
        state = new_session_state(str(uuid.uuid4()))
        save_session_state(connection, state)
        row = connection.execute(
            "SELECT updated_at FROM conversation_sessions WHERE session_id = ?",
            (state.session_id,),
        ).fetchone()
        return {
            "session_id": state.session_id,
            "created_at": row["updated_at"],
            "state": _state_response(state),
        }

    @app.get("/api/v1/sessions/{session_id}")
    def get_session(
        session_id: str,
        connection: sqlite3.Connection = Depends(database_connection),
    ) -> dict[str, Any]:
        _validate_session_id(session_id)
        row = connection.execute(
            "SELECT updated_at FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "session_not_found", "message": "Session not found."},
            )
        return {
            "session_id": session_id,
            "updated_at": row["updated_at"],
            "state": _state_response(load_session_state(connection, session_id)),
        }

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    def delete_session(
        session_id: str,
        connection: sqlite3.Connection = Depends(database_connection),
    ) -> None:
        _validate_session_id(session_id)
        with _session_operation(app, session_id):
            with connection:
                cursor = connection.execute(
                    "DELETE FROM conversation_sessions WHERE session_id = ?",
                    (session_id,),
                )
            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "code": "session_not_found",
                        "message": "Session not found.",
                    },
                )

    @app.post("/api/v1/sessions/{session_id}/messages")
    def session_message(
        session_id: str,
        body: SessionMessageRequest,
        connection: sqlite3.Connection = Depends(database_connection),
    ) -> dict[str, Any]:
        _validate_session_id(session_id)
        with _session_operation(app, session_id):
            row = connection.execute(
                "SELECT 1 FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "code": "session_not_found",
                        "message": "Session not found.",
                    },
                )
            current = load_session_state(connection, session_id)
            response, updated = _process_session_message(
                connection, current, body.message, app.state.provider
            )
            if updated is not None:
                save_session_state(connection, updated)
                response["state"] = _state_response(updated)
            else:
                response["state"] = _state_response(current)
            return response

    static_directory = Path(__file__).with_name("static")

    @app.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        return FileResponse(static_directory / "index.html")

    app.mount(
        "/assets",
        StaticFiles(directory=static_directory),
        name="frontend-assets",
    )

    return app


def _state_response(state: ConversationState) -> dict[str, Any]:
    return {
        "active_task": state.current_task,
        "active_recipe": (
            _reference_text(state.active_recipe)
            if state.active_recipe is not None
            else None
        ),
        "last_result_ids": [
            _reference_text(reference) for reference in state.previous_result_ids
        ],
        "result_limit": state.result_limit,
        "last_statistic_ingredient": state.last_statistic_ingredient,
        "filters": _jsonable(state.filters),
        "available_ingredients": list(state.available_ingredients),
        "hard_exclusions": list(state.hard_exclusions),
        "maximum_instruction_steps": state.max_instruction_steps,
        "ranking_preferences": list(state.ranking_preferences),
        "active_search": {
            "query": state.search_query,
            "field": state.search_field,
            "details_requested": state.details_requested,
        },
        "substitution_context": {
            "ingredient_to_replace": state.substitution_ingredient,
            "dietary_restrictions": list(state.dietary_restrictions),
        },
    }


def _record_matching_results(
    state: ConversationState,
    report: IngredientMatchReport,
    **updates: Any,
) -> ConversationState:
    references = tuple(
        RecipeRef(result.source, result.recipe_id) for result in report.results
    )
    return replace(
        state,
        previous_result_ids=references,
        **updates,
    )


def _rebuild_session_substitution(
    connection: sqlite3.Connection,
    state: ConversationState,
    *,
    excluded_ingredients: Sequence[str] | None = None,
    dietary_restrictions: Sequence[str] | None = None,
) -> tuple[SubstitutionProposal, ConversationState] | None:
    if state.active_recipe is None or state.substitution_ingredient is None:
        return None
    exclusions = (
        state.hard_exclusions
        if excluded_ingredients is None
        else tuple(dict.fromkeys(excluded_ingredients))
    )
    dietary = (
        state.dietary_restrictions
        if dietary_restrictions is None
        else tuple(dict.fromkeys(dietary_restrictions))
    )
    proposal = propose_substitutions(
        connection,
        state.active_recipe.source,
        state.active_recipe.recipe_id,
        state.substitution_ingredient,
        excluded_ingredients=exclusions,
        dietary_restrictions=dietary,
        preferences=state.preferences,
    )
    return proposal, replace(
        state,
        hard_exclusions=exclusions,
        dietary_restrictions=dietary,
    )


def _process_session_message(
    connection: sqlite3.Connection,
    state: ConversationState,
    message: str,
    provider: JSONProvider | None,
) -> tuple[dict[str, Any], ConversationState | None]:
    text = message.strip()
    lowered = text.casefold()

    if re.search(
        r"\b(first|second|third|1st|2nd|3rd)\s+(?:result|recipe)\b",
        lowered,
    ):
        reference, error = _lookup_reference(connection, state, text)
        if error is not None:
            return {"status": "clarification", "response": error}, None
        assert reference is not None
        detail = get_recipe(connection, reference.source, reference.recipe_id)
        assert detail is not None
        updated = replace(
            state,
            active_recipe=reference,
            current_task="recipe_detail",
        )
        return (
            {
                "status": "complete",
                "response": (
                    f"Selected {_reference_text(reference)}: "
                    f"{detail.original_record.get('title') or 'Untitled recipe'}."
                ),
                "result": {
                    "source": reference.source,
                    "recipe_id": reference.recipe_id,
                    "title": detail.original_record.get("title"),
                    "ingredients": detail.original_record.get("ingredients"),
                    "instructions": detail.original_record.get("instructions"),
                    "recipes": [
                        {
                            "source": reference.source,
                            "recipe_id": reference.recipe_id,
                            "reference": _reference_text(reference),
                            "title": detail.original_record.get("title"),
                            "ingredients": detail.original_record.get("ingredients"),
                            "instructions": detail.original_record.get("instructions"),
                        }
                    ],
                },
                "used_gemini": False,
            },
            updated,
        )

    if "original instruction" in lowered or "show its instructions" in lowered:
        reference, error = _lookup_reference(connection, state, text)
        if error is not None:
            return {"status": "clarification", "response": error}, None
        assert reference is not None
        detail = get_recipe(connection, reference.source, reference.recipe_id)
        assert detail is not None
        updated = replace(state, active_recipe=reference, current_task="recipe_detail")
        return (
            {
                "status": "complete",
                "response": str(detail.original_record.get("instructions") or ""),
                "result": {
                    "source": reference.source,
                    "recipe_id": reference.recipe_id,
                    "title": detail.original_record.get("title"),
                    "instructions": detail.original_record.get("instructions"),
                },
                "used_gemini": False,
            },
            updated,
        )

    dairy_restriction = re.search(
        r"\b(?:i\s+(?:cannot|can not|can't)\s+have|avoid|exclude|without)\s+"
        r"(?:any\s+)?dairy(?:[- ]free)?\b",
        text,
        re.IGNORECASE,
    )
    if dairy_restriction and state.current_task in {"recipe_search", "recipe_detail"}:
        if not state.search_query:
            return {
                "status": "clarification",
                "response": "Which recipe search should I update with a dairy exclusion?",
            }, None
        filters = replace(
            state.filters,
            exclude_ingredients=_merge_terms(
                state.filters.exclude_ingredients, _DAIRY_SEARCH_EXCLUSIONS
            ),
        )
        results = search_recipes(
            connection,
            state.search_query,
            field=state.search_field or "all",
            limit=state.result_limit,
            filters=filters,
        )
        updated = replace(
            state,
            current_task="recipe_search",
            filters=filters,
            previous_result_ids=tuple(
                RecipeRef(result.source, result.recipe_id) for result in results
            ),
        )
        return (
            {
                "status": "complete",
                "response": (
                    "Refined the search to exclude common dairy ingredient names. "
                    "This text-based check is not an allergy-safety guarantee."
                ),
                "result": {
                    "result_count": len(results),
                    "recipe_ids": [
                        _reference_text(RecipeRef(result.source, result.recipe_id))
                        for result in results
                    ],
                    "filters_applied": _jsonable(filters),
                },
                "used_gemini": False,
            },
            updated,
        )

    exclusion_match = re.search(
        r"\b(?:exclude|avoid|remove)\s+(.+?)(?:\s+too)?[.!?]?$",
        text,
        re.IGNORECASE,
    )
    if exclusion_match and state.current_task in {
        "recipe_search",
        "recipe_detail",
        "ingredient_matching",
        "substitution",
    }:
        term = exclusion_match.group(1).strip(" .!?")
        if state.current_task in {"recipe_search", "recipe_detail"}:
            filters = replace(
                state.filters,
                exclude_ingredients=_merge_terms(
                    state.filters.exclude_ingredients, (term,)
                ),
            )
            if not state.search_query:
                return {
                    "status": "clarification",
                    "response": "Which recipe search should I update?",
                }, None
            results = search_recipes(
                connection,
                state.search_query,
                field=state.search_field or "all",
                limit=state.result_limit,
                filters=filters,
            )
            updated = replace(
                state,
                current_task="recipe_search",
                filters=filters,
                previous_result_ids=tuple(
                    RecipeRef(result.source, result.recipe_id) for result in results
                ),
            )
            return (
                {
                    "status": "complete",
                    "response": f"Updated the search to exclude {term}.",
                    "result": {
                        "result_count": len(results),
                        "recipe_ids": [
                            _reference_text(RecipeRef(result.source, result.recipe_id))
                            for result in results
                        ],
                    },
                    "used_gemini": False,
                },
                updated,
            )
        if state.current_task == "substitution":
            rebuilt = _rebuild_session_substitution(
                connection,
                state,
                excluded_ingredients=_merge_terms(
                    state.hard_exclusions, (term,)
                ),
            )
            if rebuilt is None:
                return {
                    "status": "clarification",
                    "response": "Which active recipe substitution should I update?",
                }, None
            proposal, updated = rebuilt
            return (
                {
                    "status": "complete",
                    "response": f"Updated the substitution to exclude {term}.",
                    "result": _jsonable(proposal),
                    "used_gemini": False,
                },
                updated,
            )
        exclusions = _merge_terms(state.hard_exclusions, (term,))
        report = match_recipes(
            connection,
            state.available_ingredients,
            excluded_ingredients=exclusions,
            max_instruction_steps=state.max_instruction_steps,
            ranking_preferences=state.ranking_preferences,
            limit=state.result_limit,
        )
        updated = _record_matching_results(
            state,
            report,
            hard_exclusions=exclusions,
        )
        return (
            {
                "status": "complete",
                "response": f"Updated matching to exclude {term}.",
                "result": _jsonable(report),
                "used_gemini": False,
            },
            updated,
        )

    dietary_match = re.search(r"\b([\w-]+-free)\b", text, re.IGNORECASE)
    if dietary_match and state.current_task == "substitution":
        restriction = dietary_match.group(1).strip().lower()
        rebuilt = _rebuild_session_substitution(
            connection,
            state,
            dietary_restrictions=(*state.dietary_restrictions, restriction),
        )
        if rebuilt is None:
            return {
                "status": "clarification",
                "response": "Which active recipe substitution should I update?",
            }, None
        proposal, updated = rebuilt
        return (
            {
                "status": "complete",
                "response": f"Updated the substitution restriction to {restriction}.",
                "result": _jsonable(proposal),
                "used_gemini": False,
            },
            updated,
        )

    preference_updates: list[str] = []
    if re.search(r"\bfewer missing ingredients\b", lowered):
        preference_updates.append("fewer_missing")
    if re.search(r"\bfewer additional ingredients\b", lowered):
        preference_updates.append("fewer_additional")
    if (
        re.search(r"\bfewer steps\b|\bfewer instruction steps\b", lowered)
        and not re.search(r"\bshow only\b.*\bfewer (?:instruction )?steps\b", lowered)
    ):
        preference_updates.append("fewer_steps")
    if preference_updates:
        if state.current_task != "ingredient_matching" or not state.available_ingredients:
            return {
                "status": "clarification",
                "response": "Please run an ingredient match before changing its ranking preference.",
            }, None
        preferences = normalize_ranking_preferences(
            (*state.ranking_preferences, *preference_updates)
        )
        report = match_recipes(
            connection,
            state.available_ingredients,
            excluded_ingredients=state.hard_exclusions,
            max_instruction_steps=state.max_instruction_steps,
            ranking_preferences=preferences,
            limit=state.result_limit,
        )
        updated = _record_matching_results(
            state,
            report,
            ranking_preferences=preferences,
        )
        labels = {
            "fewer_missing": "fewer missing ingredients",
            "fewer_additional": "fewer additional ingredients",
            "fewer_steps": "fewer instruction steps",
        }
        return (
            {
                "status": "complete",
                "response": (
                    "Updated ranking to prefer "
                    + ", then ".join(labels[item] for item in preferences)
                    + "."
                ),
                "result": _jsonable(report),
                "used_gemini": False,
            },
            updated,
        )

    if re.search(r"\bshow only\b.*\bfewer (?:instruction )?steps\b", lowered):
        if state.current_task != "ingredient_matching" or not state.previous_result_ids:
            return {
                "status": "clarification",
                "response": "Run an ingredient match first so I can compare its instruction-step count.",
            }, None
        step_counts: list[int] = []
        for reference in state.previous_result_ids:
            row = connection.execute(
                """
                SELECT instruction_step_count FROM recipes
                WHERE source = ? AND recipe_id = ?
                """,
                (reference.source, reference.recipe_id),
            ).fetchone()
            if row is not None and isinstance(row["instruction_step_count"], int):
                step_counts.append(row["instruction_step_count"])
        if not step_counts or max(step_counts) <= 1:
            return {
                "status": "clarification",
                "response": "The previous results do not establish a lower step limit.",
            }, None
        max_steps = max(step_counts) - 1
        report = match_recipes(
            connection,
            state.available_ingredients,
            excluded_ingredients=state.hard_exclusions,
            max_instruction_steps=max_steps,
            ranking_preferences=state.ranking_preferences,
            limit=state.result_limit,
        )
        updated = _record_matching_results(
            state,
            report,
            max_instruction_steps=max_steps,
        )
        return (
            {
                "status": "complete",
                "response": (
                    f"Filtered matching results to at most {max_steps} "
                    "instruction steps."
                ),
                "result": _jsonable(report),
                "used_gemini": False,
            },
            updated,
        )

    if re.search(r"\breplace\b", lowered):
        ingredient_match = re.search(
            r"\breplace\s+(.+?)\s+in\b", text, re.IGNORECASE
        )
        if ingredient_match is None:
            return {
                "status": "clarification",
                "response": "Which ingredient should I replace?",
            }, None
        reference, error = _lookup_reference(connection, state, text)
        if error is not None:
            return {"status": "clarification", "response": error}, None
        assert reference is not None
        ingredient = ingredient_match.group(1).strip()
        dietary_restrictions = (
            state.dietary_restrictions
            if state.current_task == "substitution"
            else ()
        )
        if re.search(r"\bwithout\s+dairy\b|\bdairy[- ]free\b", lowered):
            dietary_restrictions = _merge_terms(
                dietary_restrictions, ("dairy-free",)
            )
        proposal = propose_substitutions(
            connection,
            reference.source,
            reference.recipe_id,
            ingredient,
            excluded_ingredients=state.hard_exclusions,
            dietary_restrictions=dietary_restrictions,
            preferences=state.preferences,
        )
        updated = replace(
            state,
            active_recipe=reference,
            current_task="substitution",
            substitution_ingredient=ingredient,
            dietary_restrictions=dietary_restrictions,
        )
        return (
            {
                "status": "complete",
                "response": (
                    f"Substitution evidence for {ingredient} in "
                    f"{_reference_text(reference)}."
                ),
                "result": _jsonable(proposal),
                "used_gemini": False,
            },
            updated,
        )

    percentage_followup = (
        state.current_task == "recipe_search"
        and state.last_statistic_ingredient is not None
        and re.search(r"\b(?:percentage|percent|what %)\b", lowered) is not None
    )
    if percentage_followup:
        request = _request_defaults(
            "recipe_search",
            statistic="ingredient_prevalence",
            statistic_ingredient=state.last_statistic_ingredient,
            sources=state.filters.sources,
            dish_name=state.filters.title_query,
            include_ingredients=state.filters.include_ingredients,
            exclude_ingredients=state.filters.exclude_ingredients,
        )
    elif provider is not None:
        try:
            resolution = interpret_request(text, provider)
        except LLMError as error:
            logger.warning(
                "Session request interpretation unavailable (%s)",
                type(error).__name__,
            )
            resolution = None
        if resolution is not None:
            request = resolution.request
            if request is None:
                return {
                    "status": "clarification",
                    "response": resolution.clarification_question
                    or "Please clarify your request.",
                    "used_gemini": False,
                }, None
        else:
            request = _parse_fallback_request(text, state)
    else:
        request = _parse_fallback_request(text, state)
    if request is None:
        return {
            "status": "clarification",
            "response": (
                "I could not interpret that without Gemini. Try a simple search, "
                "an ingredient list beginning with 'I have', or a supported "
                "follow-up."
            ),
            "used_gemini": False,
        }, None

    if request.task == "recipe_search":
        same_task = state.current_task == "recipe_search"
        request = replace(
            request,
            include_ingredients=tuple(
                dict.fromkeys(
                    (*state.filters.include_ingredients, *request.include_ingredients)
                )
            )
            if same_task
            else request.include_ingredients,
            exclude_ingredients=tuple(
                dict.fromkeys(
                    (*state.filters.exclude_ingredients, *request.exclude_ingredients)
                )
            )
            if same_task
            else request.exclude_ingredients,
            sources=request.sources or (state.filters.sources if same_task else ()),
        )
    elif request.task == "ingredient_matching":
        same_task = state.current_task == "ingredient_matching"
        request = replace(
            request,
            exclude_ingredients=tuple(
                dict.fromkeys((*state.hard_exclusions, *request.exclude_ingredients))
            )
            if same_task
            else request.exclude_ingredients,
            max_instruction_steps=(
                request.max_instruction_steps or state.max_instruction_steps
            )
            if same_task
            else request.max_instruction_steps,
            ranking_preferences=request.ranking_preferences
            or (state.ranking_preferences if same_task else ()),
            available_ingredients=request.available_ingredients
            or (state.available_ingredients if same_task else ()),
        )
        request = replace(
            request,
            ranking_preferences=normalize_ranking_preferences(
                request.ranking_preferences
            ),
        )
    elif request.task == "substitution" and request.recipe_reference_text is None:
        if state.active_recipe is not None:
            request = replace(
                request,
                recipe_reference_text=_reference_text(state.active_recipe),
            )

    execution = execute_request(connection, request)
    grounded = generate_grounded_response(execution, provider)
    updated = state
    if request.task == "recipe_search":
        refs = tuple(
            RecipeRef(item["source"], item["recipe_id"])
            for item in execution.payload.get("recipes", ())
        )
        updated = replace(
            state,
            current_task="recipe_search",
            filters=RecipeFilters(
                sources=request.sources,
                title_query=request.dish_name,
                include_ingredients=request.include_ingredients,
                exclude_ingredients=request.exclude_ingredients,
            ),
            search_query=request.search_query or request.dish_name,
            search_field=request.search_field or "all",
            details_requested=request.details_requested,
            previous_result_ids=refs,
            active_recipe=None,
            result_limit=request.limit,
            last_statistic_ingredient=(
                request.statistic_ingredient
                if request.statistic == "ingredient_prevalence"
                else None
            ),
            available_ingredients=(),
            max_instruction_steps=None,
            ranking_preferences=(),
            hard_exclusions=(),
            preferences=(),
            dietary_restrictions=(),
            substitution_ingredient=None,
        )
    elif request.task == "ingredient_matching":
        refs = tuple(
            RecipeRef(result["source"], result["recipe_id"])
            for result in execution.payload["report"]["results"]
        )
        updated = replace(
            state,
            result_limit=request.limit,
            current_task="ingredient_matching",
            active_recipe=None,
            filters=state.filters if same_task else RecipeFilters(),
            available_ingredients=request.available_ingredients,
            hard_exclusions=request.exclude_ingredients,
            max_instruction_steps=request.max_instruction_steps,
            ranking_preferences=request.ranking_preferences,
            previous_result_ids=refs,
            search_query=state.search_query if same_task else None,
            search_field=state.search_field if same_task else None,
            details_requested=state.details_requested if same_task else False,
            preferences=(),
            dietary_restrictions=(),
            substitution_ingredient=None,
        )
    elif request.task == "substitution":
        proposal = execution.payload.get("proposal")
        if proposal is not None:
            ref = proposal["original_recipe"]
            updated = replace(
                state,
                active_recipe=RecipeRef(ref["source"], ref["recipe_id"]),
                current_task="substitution",
                previous_result_ids=(
                    RecipeRef(ref["source"], ref["recipe_id"]),
                ),
                filters=RecipeFilters(),
                available_ingredients=(),
                max_instruction_steps=None,
                ranking_preferences=(),
                search_query=None,
                search_field=None,
                details_requested=False,
                substitution_ingredient=request.ingredient_to_replace,
                hard_exclusions=request.exclude_ingredients,
                dietary_restrictions=request.dietary_restrictions,
                preferences=request.preferences,
            )
    return (
        {
            "status": execution.status,
            "response": grounded.text,
            "used_gemini": grounded.used_gemini,
            "result": _jsonable(execution.payload),
            "clarification_question": execution.clarification_question,
        },
        updated,
    )


app = create_app()
