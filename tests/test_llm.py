from __future__ import annotations

import json
import sqlite3
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai.errors import APIError

from recipe_intelligence.assistant import (
    FACT_SELECTION_SCHEMA,
    INTERPRETATION_SCHEMA,
    ExecutionResult,
    GroundedFact,
    execute_request,
    generate_grounded_response,
    handle_natural_language,
    interpret_request,
    parse_request_interpretation,
)
from recipe_intelligence.database import connect_database, ingest_dataset_archive
from recipe_intelligence.ingestion import DATASET_FILES
from recipe_intelligence.llm import (
    GeminiProvider,
    GeminiSettings,
    LLMRateLimited,
    LLMTimeout,
    MalformedLLMResponse,
    MissingAPIKey,
    provider_from_environment,
)


class StaticProvider:
    def __init__(
        self,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_json(
        self,
        *,
        system_instruction: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "payload": payload,
                "schema": schema,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class FakeModels:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = 0
        self.arguments: dict[str, Any] | None = None

    def generate_content(self, **kwargs: Any) -> object:
        self.calls += 1
        self.arguments = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def fixture_records() -> dict[str, dict[str, object]]:
    return {
        "garlic-soup": {
            "title": "Garlic Soup",
            "ingredients": ["garlic", "onion", "vegetable broth"],
            "instructions": "Simmer garlic and onion in broth.",
        },
        "garlic-bread": {
            "title": "Garlic Bread",
            "ingredients": ["garlic", "bread", "butter"],
            "instructions": "Mix garlic and butter. Bake on bread.",
        },
        "peanut-garlic": {
            "title": "Peanut Garlic Spread",
            "ingredients": ["garlic", "peanut butter", "onion"],
            "instructions": "Mix and serve.",
        },
        "milk-cake": {
            "title": "Vanilla Breakfast Cake",
            "ingredients": ["1 cup milk", "2 cups flour", "1 cup sugar", "2 eggs"],
            "instructions": "1. Mix milk, flour, sugar, and eggs.\n2. Bake.",
        },
        "oat-milk-cake": {
            "title": "Vanilla Breakfast Cake with Oat Milk",
            "ingredients": [
                "1 cup oat milk",
                "2 cups flour",
                "1 cup sugar",
                "2 eggs",
            ],
            "instructions": (
                "1. Mix oat milk, flour, sugar, and eggs.\n2. Bake."
            ),
        },
        "turmeric-soup": {
            "title": "Turmeric Lentil Soup",
            "ingredients": ["turmeric", "lentils", "vegetable broth"],
            "instructions": "Simmer until tender.",
        },
    }


@pytest.fixture
def connection(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    database = connect_database(":memory:")
    archive_path = tmp_path / "assistant-recipes.zip"
    records = fixture_records()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for filename, source in DATASET_FILES.items():
            archive.writestr(
                filename,
                json.dumps(records if source == "ar" else {}),
            )
    ingest_dataset_archive(database, archive_path)
    yield database
    database.close()


def request_payload(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task": "unsupported",
        "search_query": None,
        "dish_name": None,
        "search_field": None,
        "include_ingredients": [],
        "exclude_ingredients": [],
        "sources": [],
        "details_requested": False,
        "statistic": None,
        "statistic_ingredient": None,
        "available_ingredients": [],
        "max_instruction_steps": None,
        "ranking_preferences": [],
        "recipe_reference_text": None,
        "ingredient_to_replace": None,
        "replacement_ingredient": None,
        "dietary_restrictions": [],
        "preferences": [],
        "limit": 5,
    }
    result.update(overrides)
    return result


def selected_all_facts(execution: ExecutionResult, opening: str = "results") -> dict[str, Any]:
    return {
        "opening": opening,
        "fact_ids": [fact.fact_id for fact in execution.facts],
    }


def test_search_interpretation_is_typed_and_entities_are_user_grounded() -> None:
    message = "Find chicken soup recipes without peanut and show details"
    payload = request_payload(
        task="recipe_search",
        search_query="chicken soup",
        dish_name="chicken soup",
        search_field="title",
        exclude_ingredients=["peanut"],
        details_requested=True,
    )

    result = parse_request_interpretation(payload, message)

    assert result.request is not None
    assert result.request.task == "recipe_search"
    assert result.request.exclude_ingredients == ("peanut",)
    assert result.request.details_requested
    assert INTERPRETATION_SCHEMA["additionalProperties"] is False


def test_matching_and_substitution_interpretation_extracts_constraints() -> None:
    matching_message = "I have garlic and onion, avoid peanut, no more than 3 steps"
    matching = parse_request_interpretation(
        request_payload(
            task="ingredient_matching",
            available_ingredients=["garlic", "onion"],
            exclude_ingredients=["peanut"],
            max_instruction_steps=3,
        ),
        matching_message,
    )
    substitution_message = (
        "Replace milk in Vanilla Breakfast Cake ar:milk-cake with oat milk "
        "and make it dairy-free"
    )
    substitution = parse_request_interpretation(
        request_payload(
            task="substitution",
            recipe_reference_text="ar:milk-cake",
            ingredient_to_replace="milk",
            replacement_ingredient="oat milk",
            dietary_restrictions=["dairy-free"],
        ),
        substitution_message,
    )

    assert matching.request is not None
    assert matching.request.max_instruction_steps == 3
    assert matching.request.exclude_ingredients == ("peanut",)
    assert substitution.request is not None
    assert substitution.request.recipe_reference_text == "ar:milk-cake"


def test_invalid_parameters_and_ambiguous_requests_are_rejected_or_clarified() -> None:
    message = "Find garlic soup"
    with pytest.raises(ValueError, match="max_instruction_steps"):
        parse_request_interpretation(
            request_payload(
                task="ingredient_matching",
                available_ingredients=["garlic"],
                max_instruction_steps="three",
            ),
            message,
        )
    with pytest.raises(ValueError, match="unexpected fields"):
        parse_request_interpretation(
            {**request_payload(), "recipe_id": "invented-id"},
            message,
        )
    ambiguous = parse_request_interpretation(
        request_payload(task="substitution"),
        "Please change something in a recipe",
    )
    assert ambiguous.request is None
    assert ambiguous.clarification_question == (
        "Which recipe title or recipe ID should I use?"
    )
    with pytest.raises(ValueError, match="verbatim"):
        parse_request_interpretation(
            request_payload(
                task="recipe_search",
                search_query="invented carbonara",
            ),
            message,
        )
    with pytest.raises(ValueError, match="verbatim"):
        parse_request_interpretation(
            request_payload(
                task="recipe_search",
                search_query="ham",
            ),
            "Find hamster recipes",
        )


def test_interpret_request_uses_structured_provider_without_database_access() -> None:
    provider = StaticProvider(
        request_payload(
            task="ingredient_matching",
            available_ingredients=["garlic"],
        )
    )

    resolution = interpret_request("I have garlic", provider)

    assert resolution.request is not None
    assert resolution.request.available_ingredients == ("garlic",)
    assert provider.calls[0]["schema"] is INTERPRETATION_SCHEMA
    assert provider.calls[0]["payload"] == {"user_request": "I have garlic"}


def test_search_response_uses_retrieved_records_and_original_details(
    connection: sqlite3.Connection,
) -> None:
    request = parse_request_interpretation(
        request_payload(
            task="recipe_search",
            search_query="garlic bread",
            dish_name="garlic bread",
            search_field="title",
            details_requested=True,
        ),
        "Find garlic bread and show details",
    ).request
    assert request is not None
    execution = execute_request(connection, request)
    response = generate_grounded_response(
        execution, StaticProvider(selected_all_facts(execution))
    )

    assert execution.payload["recipes"][0]["recipe_id"] == "garlic-bread"
    assert execution.payload["recipes"][0]["original_record"]["ingredients"] == [
        "garlic",
        "bread",
        "butter",
    ]
    assert "ar:garlic-bread" in response.text
    assert "Mix garlic and butter. Bake on bread." in response.text
    assert response.used_gemini


def test_statistics_response_preserves_computed_value_method_and_denominator(
    connection: sqlite3.Connection,
) -> None:
    request = parse_request_interpretation(
        request_payload(task="recipe_search", statistic="recipe_count"),
        "How many recipes are in the dataset?",
    ).request
    assert request is not None
    execution = execute_request(connection, request)
    response = generate_grounded_response(
        execution, StaticProvider(selected_all_facts(execution, "calculation"))
    )

    statistic = execution.payload["statistics"][0]
    assert statistic["value"] == 6
    assert statistic["denominator"] == 6
    assert f"numerator {statistic['numerator']}" in response.text
    assert "Count every recipe record matching" in response.text


def test_matching_response_preserves_ranked_results_and_hard_exclusions(
    connection: sqlite3.Connection,
) -> None:
    request = parse_request_interpretation(
        request_payload(
            task="ingredient_matching",
            available_ingredients=["garlic", "bread"],
            exclude_ingredients=["peanut"],
            max_instruction_steps=3,
        ),
        "I have garlic and bread; avoid peanut; max 3 steps",
    ).request
    assert request is not None
    execution = execute_request(connection, request)
    response = generate_grounded_response(
        execution, StaticProvider(selected_all_facts(execution))
    )

    report = execution.payload["report"]
    assert report["results"]
    assert "peanut-garlic" not in {
        item["recipe_id"] for item in report["results"]
    }
    positions = [
        response.text.index(f"Rank {index}:")
        for index in range(1, len(report["results"]) + 1)
    ]
    assert positions == sorted(positions)
    assert "sufficiency verified: False" in response.text


def test_substitution_response_preserves_weak_evidence_and_refusal(
    connection: sqlite3.Connection,
) -> None:
    request = parse_request_interpretation(
        request_payload(
            task="substitution",
            recipe_reference_text="Vanilla Breakfast Cake",
            ingredient_to_replace="milk",
            replacement_ingredient="turmeric",
            exclude_ingredients=["peanut"],
        ),
        "Replace milk in Vanilla Breakfast Cake with turmeric; avoid peanut",
    ).request
    assert request is not None
    execution = execute_request(connection, request)
    response = generate_grounded_response(
        execution, StaticProvider(selected_all_facts(execution, "evidence"))
    )

    option = execution.payload["proposal"]["options"][0]
    assert option["evidence_strength"] == "weak"
    assert option["sufficiently_supported"] is False
    assert option["revised_ingredient_list"] is None
    assert "evidence weak" in response.text
    assert "sufficiently supported False" in response.text
    assert "ar:turmeric-soup" in response.text


def test_invalid_ids_or_unsupported_claim_fields_fall_back_to_app_owned_facts(
    connection: sqlite3.Connection,
) -> None:
    request = parse_request_interpretation(
        request_payload(
            task="recipe_search",
            search_query="garlic bread",
            search_field="title",
        ),
        "Find garlic bread",
    ).request
    assert request is not None
    execution = execute_request(connection, request)
    invalid_id = StaticProvider(
        {"opening": "results", "fact_ids": ["invented-recipe-id"]}
    )
    unsupported_claim = StaticProvider(
        {
            "opening": "results",
            "fact_ids": [fact.fact_id for fact in execution.facts],
            "text": "This has 20 grams of protein and takes 10 minutes.",
        }
    )

    rejected_id = generate_grounded_response(execution, invalid_id)
    rejected_claim = generate_grounded_response(execution, unsupported_claim)

    assert not rejected_id.used_gemini
    assert "invented-recipe-id" not in rejected_id.text
    assert not rejected_claim.used_gemini
    assert "20 grams" not in rejected_claim.text
    assert rejected_claim.fallback_reason == "MalformedLLMResponse"


def test_missing_api_key_and_disabled_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    with pytest.raises(MissingAPIKey):
        provider_from_environment()

    monkeypatch.setenv("LLM_ENABLED", "false")
    assert provider_from_environment() is None


def test_timeout_rate_limit_malformed_and_oversized_sdk_responses() -> None:
    settings = GeminiSettings(
        enabled=True, api_key="not-a-real-key", model="test-model"
    )
    timeout_models = FakeModels(error=httpx.ReadTimeout("no response"))
    with pytest.raises(LLMTimeout):
        GeminiProvider(settings, SimpleNamespace(models=timeout_models)).generate_json(
            system_instruction="test",
            payload={"input": "test"},
            schema=FACT_SELECTION_SCHEMA,
        )
    assert timeout_models.calls == 1

    rate_limited = FakeModels(
        error=APIError(429, {"error": {"message": "rate limited"}})
    )
    with pytest.raises(LLMRateLimited):
        GeminiProvider(settings, SimpleNamespace(models=rate_limited)).generate_json(
            system_instruction="test",
            payload={"input": "test"},
            schema=FACT_SELECTION_SCHEMA,
        )
    assert rate_limited.calls == 1

    for text in (None, "not json", json.dumps(["not", "an", "object"])):
        models = FakeModels(response=SimpleNamespace(text=text))
        with pytest.raises(MalformedLLMResponse):
            GeminiProvider(
                settings, SimpleNamespace(models=models)
            ).generate_json(
                system_instruction="test",
                payload={"input": "test"},
                schema=FACT_SELECTION_SCHEMA,
            )

    settings = GeminiSettings(
        enabled=True,
        api_key="not-a-real-key",
        model="test-model",
        max_response_characters=4,
    )
    models = FakeModels(response=SimpleNamespace(text='{"ok": true}'))
    with pytest.raises(MalformedLLMResponse, match="size limit"):
        GeminiProvider(settings, SimpleNamespace(models=models)).generate_json(
            system_instruction="test",
            payload={"input": "test"},
            schema=FACT_SELECTION_SCHEMA,
        )


def test_gemini_sdk_request_is_structured_bounded_and_not_retried() -> None:
    models = FakeModels(response=SimpleNamespace(text='{"ok": true}'))
    provider = GeminiProvider(
        GeminiSettings(
            enabled=True,
            api_key="not-a-real-key",
            model="test-model",
        ),
        client=SimpleNamespace(models=models),
    )

    assert provider.generate_json(
        system_instruction="stay grounded",
        payload={"evidence": "untrusted"},
        schema={"type": "object"},
    ) == {"ok": True}
    assert models.calls == 1
    assert models.arguments is not None
    assert models.arguments["model"] == "test-model"
    assert models.arguments["config"].response_mime_type == "application/json"
    assert models.arguments["config"].response_json_schema == {"type": "object"}


def test_disabled_natural_language_and_deterministic_service_remain_available(
    connection: sqlite3.Connection,
) -> None:
    disabled = handle_natural_language(connection, "find garlic bread", None)
    request = parse_request_interpretation(
        request_payload(
            task="recipe_search",
            search_query="garlic bread",
            search_field="title",
        ),
        "find garlic bread",
    ).request
    assert request is not None

    execution = execute_request(connection, request)

    assert disabled.status == "unavailable"
    assert disabled.execution is None
    assert execution.status == "complete"
    assert execution.payload["recipes"][0]["recipe_id"] == "garlic-bread"
    assert generate_grounded_response(execution, None).used_gemini is False


def test_interpreter_and_response_provider_failures_are_non_fabricating(
    connection: sqlite3.Connection,
) -> None:
    failing = StaticProvider(error=LLMTimeout("timeout"))
    result = handle_natural_language(connection, "find garlic bread", failing)
    assert result.status == "unavailable"
    assert result.execution is None

    execution = ExecutionResult(
        task="recipe_search",
        status="complete",
        payload={},
        facts=(GroundedFact("fact-1", "Only one verified fact."),),
    )
    failed_response = generate_grounded_response(
        execution, StaticProvider(error=LLMTimeout("timeout"))
    )
    assert not failed_response.used_gemini
    assert failed_response.text == "Only one verified fact."
    assert failed_response.fallback_reason == "LLMTimeout"


def test_interpreter_rejects_malformed_provider_output() -> None:
    provider = StaticProvider({"task": "recipe_search"})
    with pytest.raises(MalformedLLMResponse):
        interpret_request("find garlic", provider)
