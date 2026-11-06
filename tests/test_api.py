from __future__ import annotations

import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recipe_intelligence.api import (
    _parse_fallback_request,
    create_app,
)
from recipe_intelligence.database import connect_database, ingest_dataset_archive
from recipe_intelligence.ingestion import DATASET_FILES
from recipe_intelligence.conversation import new_session_state
from recipe_intelligence.retrieval import get_recipe


def fixture_records() -> dict[str, dict[str, object]]:
    return {
        "garlic-bread-egg": {
            "title": "Garlic Bread with Egg",
            "ingredients": ["garlic", "bread", "egg", "butter"],
            "instructions": "1. Mix garlic, egg, and butter.\n2. Bake on bread.",
        },
        "garlic-bread-cream": {
            "title": "Creamy Garlic Bread",
            "ingredients": ["garlic", "bread", "cream", "butter"],
            "instructions": "1. Mix cream, garlic, and butter.\n2. Bake on bread.",
        },
        "garlic-loaf": {
            "title": "Garlic Loaf",
            "ingredients": ["garlic", "bread", "butter"],
            "instructions": "1. Mix garlic and butter.\n2. Bake on bread.",
        },
        "garlic-soup": {
            "title": "Garlic Soup",
            "ingredients": ["garlic", "onion", "vegetable broth"],
            "instructions": "1. Saute garlic and onion.\n2. Simmer in broth.",
        },
        "garlic-onion": {
            "title": "Garlic Onion",
            "ingredients": ["garlic", "onion"],
            "instructions": "1. Chop onion.\n2. Cook with garlic.",
        },
        "garlic-many": {
            "title": "Garlic Many Ingredients",
            "ingredients": ["garlic", "onion", "bread", "salt"],
            "instructions": "1. Mix everything.\n2. Bake.",
        },
        "peanut-garlic": {
            "title": "Peanut Garlic Spread",
            "ingredients": ["garlic", "peanut butter", "onion"],
            "instructions": "Mix and serve.",
        },
        "oat-cream-bread": {
            "title": "Oat Cream Bread",
            "ingredients": ["oat cream", "bread", "garlic", "butter"],
            "instructions": "1. Mix oat cream, bread, garlic, butter.\n2. Bake.",
        },
        "soup/variant": {
            "title": "Garlic Soup Variant",
            "ingredients": ["garlic", "broth"],
            "instructions": "Simmer.",
        },
    }


def make_archive(path: Path) -> Path:
    records = fixture_records()
    with zipfile.ZipFile(path, "w") as archive:
        for filename, source in DATASET_FILES.items():
            source_records = (
                records
                if source == "ar"
                else {"garlic-soup": records["garlic-soup"]}
                if source == "epi"
                else {}
            )
            archive.writestr(
                filename,
                json.dumps(source_records),
            )
    return path


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    database_path = tmp_path / "api-recipes.sqlite3"
    connection = connect_database(database_path)
    ingest_dataset_archive(connection, make_archive(tmp_path / "api-recipes.zip"))
    connection.close()
    app = create_app(database_path, use_environment_provider=False)
    with TestClient(app) as client:
        yield client, database_path


def test_health_and_session_create_retrieve_and_persist_after_restart(
    api: tuple[TestClient, Path],
) -> None:
    client, database_path = api
    health = client.get("/health")
    created = client.post("/api/v1/sessions")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "database": "ok",
        "gemini": "disabled",
    }
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    assert client.get(f"/api/v1/sessions/{session_id}").status_code == 200

    search = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Find garlic bread"},
    )
    assert search.status_code == 200
    updated = search.json()["state"]
    assert updated["active_task"] == "recipe_search"
    assert len(updated["last_result_ids"]) == 5

    restarted_client = TestClient(create_app(database_path, use_environment_provider=False))
    restored = restarted_client.get(f"/api/v1/sessions/{session_id}")
    assert restored.status_code == 200
    assert restored.json()["state"] == updated
    restarted_client.close()


def test_frontend_assets_are_served_from_the_api_origin(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    page = client.get("/")
    script = client.get("/assets/app.js")
    stylesheet = client.get("/assets/styles.css")

    assert page.status_code == script.status_code == stylesheet.status_code == 200
    assert 'id="message-form"' in page.text
    assert "api/v1/sessions/" in script.text
    assert "recipe-card" in stylesheet.text


def test_session_not_found_and_invalid_id_are_reported(api: tuple[TestClient, Path]) -> None:
    client, _ = api
    missing_id = "00000000-0000-4000-8000-000000000001"

    missing = client.get(f"/api/v1/sessions/{missing_id}")
    invalid = client.get("/api/v1/sessions/not-a-uuid")
    message_missing = client.post(
        f"/api/v1/sessions/{missing_id}/messages",
        json={"message": "Find garlic"},
    )

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "session_not_found"
    assert invalid.status_code == 400
    assert message_missing.status_code == 404


def test_session_can_be_deleted_explicitly(api: tuple[TestClient, Path]) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]

    deleted = client.delete(f"/api/v1/sessions/{session_id}")
    missing = client.get(f"/api/v1/sessions/{session_id}")

    assert deleted.status_code == 204
    assert missing.status_code == 404


def test_recipe_details_support_stable_ids_containing_slashes(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    response = client.get("/api/v1/recipes/soup%2Fvariant?source=ar")
    assert response.status_code == 200
    assert response.json()["recipe_id"] == "soup/variant"


def test_ambiguous_stable_ids_require_a_source(api: tuple[TestClient, Path]) -> None:
    client, _ = api
    details = client.get("/api/v1/recipes/garlic-soup")
    substitution = client.post(
        "/api/v1/substitutions",
        json={"recipe_id": "garlic-soup", "ingredient_to_replace": "garlic"},
    )
    assert details.status_code == 409
    assert substitution.status_code == 409


def test_search_details_analytics_matching_and_substitution_endpoints(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    search = client.post(
        "/api/v1/recipes/search",
        json={
            "query": "garlic bread",
            "field": "title",
            "include_details": True,
            "filters": {"exclude_ingredients": ["egg"]},
        },
    )
    details = client.get("/api/v1/recipes/garlic-bread-cream?source=ar")
    not_found = client.get("/api/v1/recipes/not-here?source=ar")
    stats = client.post(
        "/api/v1/analytics",
        json={"statistic": "recipe_count", "filters": {"sources": ["ar"]}},
    )
    match = client.post(
        "/api/v1/matching",
        json={
            "available_ingredients": ["garlic", "onion", "bread"],
            "excluded_ingredients": ["peanut"],
            "ranking_preferences": ["fewer_additional"],
        },
    )
    substitution = client.post(
        "/api/v1/substitutions",
        json={
            "recipe_id": "garlic-bread-cream",
            "source": "ar",
            "ingredient_to_replace": "cream",
            "replacement_ingredients": ["oat cream"],
            "dietary_restrictions": ["dairy-free"],
        },
    )

    assert search.status_code == 200
    assert search.json()["filters_applied"]["exclude_ingredients"] == ["egg"]
    assert all("egg" not in recipe["recipe_id"] for recipe in search.json()["results"])
    assert search.json()["results"][0]["instructions"]
    assert details.json()["recipe"]["title"] == "Creamy Garlic Bread"
    assert not_found.status_code == 404
    assert stats.json()["result"]["value"] == len(fixture_records())
    report = match.json()["report"]
    assert report["ranking_preferences"] == ["fewer_additional"]
    assert report["quantity_sufficiency_verified"] is False
    proposal = substitution.json()["proposal"]
    assert proposal["original_recipe"]["recipe_id"] == "garlic-bread-cream"
    assert proposal["options"][0]["replacement_ingredient"] == "oat cream"
    assert proposal["options"][0]["sufficiently_supported"] is False
    assert proposal["options"][0]["revised_ingredient_list"] is None


def test_matching_preference_changes_order_without_changing_eligibility(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    baseline = client.post(
        "/api/v1/matching",
        json={
            "available_ingredients": ["garlic", "onion", "bread"],
            "excluded_ingredients": ["peanut"],
        },
    ).json()["report"]
    preferred = client.post(
        "/api/v1/matching",
        json={
            "available_ingredients": ["garlic", "onion", "bread"],
            "excluded_ingredients": ["peanut"],
            "ranking_preferences": ["fewer_additional"],
        },
    ).json()["report"]

    baseline_ids = [item["recipe_id"] for item in baseline["results"]]
    preferred_ids = [item["recipe_id"] for item in preferred["results"]]
    assert set(baseline_ids) == set(preferred_ids)
    assert baseline_ids[0] != preferred_ids[0]
    assert "peanut-garlic" not in preferred_ids
    assert preferred["ranking_method"].startswith("Apply preferences")


def test_multiturn_selection_original_instructions_and_exclusion_followup(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    found = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Find garlic bread"},
    ).json()
    refs = found["state"]["last_result_ids"]
    assert len(refs) >= 2

    selected = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Show the second recipe."},
    ).json()
    assert selected["state"]["active_recipe"] == refs[1]
    assert selected["result"]["recipes"][0]["recipe_id"] == refs[1].split(":", 1)[1]
    assert selected["result"]["recipes"][0]["ingredients"]

    instructions = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Show its original instructions."},
    ).json()
    connection = connect_database(api[1])
    source, recipe_id = refs[1].split(":", 1)
    actual = get_recipe(connection, source, recipe_id)
    connection.close()
    assert instructions["result"]["instructions"] == actual.original_record["instructions"]

    restricted = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Exclude egg too."},
    ).json()
    assert restricted["state"]["filters"]["exclude_ingredients"] == ["egg"]
    assert all("egg" not in result.split(":", 1)[1] for result in restricted["result"]["recipe_ids"])


def test_active_recipe_reference_supports_substitution_and_step_filter_followups(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    search = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Find creamy garlic bread"},
    ).json()
    assert search["state"]["last_result_ids"] == ["ar:garlic-bread-cream"]

    substitution = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Replace cream in that recipe."},
    ).json()
    assert substitution["status"] == "complete"
    assert substitution["result"]["original_recipe"]["recipe_id"] == "garlic-bread-cream"
    restricted = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Exclude peanut too."},
    ).json()
    assert restricted["state"]["hard_exclusions"] == ["peanut"]
    assert all(
        "peanut" not in " ".join(option["revised_ingredient_list"] or [])
        for option in restricted["result"]["options"]
    )
    dairy_free = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Make it dairy-free."},
    ).json()
    assert dairy_free["state"]["substitution_context"]["dietary_restrictions"] == [
        "dairy-free"
    ]

    matching = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "I have garlic, bread and butter"},
    ).json()
    assert matching["state"]["active_task"] == "ingredient_matching"
    fewer_steps = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Show only recipes with fewer steps."},
    ).json()
    assert fewer_steps["status"] == "complete"
    assert fewer_steps["state"]["maximum_instruction_steps"] is not None
    assert fewer_steps["result"]["max_instruction_steps"] == fewer_steps["state"]["maximum_instruction_steps"]


def test_followup_preferences_and_ambiguous_reference_do_not_corrupt_state(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    before_match = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Find garlic bread"},
    ).json()
    ambiguous = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Replace cream in that recipe."},
    ).json()
    assert ambiguous["status"] == "clarification"
    assert ambiguous["state"] == before_match["state"]

    match = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "I have garlic, onion and bread"},
    ).json()
    assert match["state"]["active_task"] == "ingredient_matching"
    preferred = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Prefer fewer additional ingredients"},
    ).json()
    assert preferred["state"]["ranking_preferences"] == ["fewer_additional"]
    assert preferred["result"]["ranking_preferences"] == ["fewer_additional"]
    assert preferred["state"]["last_result_ids"] == [
        f"{item['source']}:{item['recipe_id']}"
        for item in preferred["result"]["results"]
    ]


def test_matching_hard_exclusion_survives_preference_followup(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "I have garlic, onion and bread"},
    )
    excluded = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Exclude peanut too."},
    ).json()
    preferred = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Prefer fewer additional ingredients"},
    ).json()

    assert excluded["state"]["hard_exclusions"] == ["peanut"]
    assert preferred["state"]["hard_exclusions"] == ["peanut"]
    assert preferred["result"]["excluded_ingredients"] == ["peanut"]
    assert "peanut-garlic" not in {
        item["recipe_id"] for item in preferred["result"]["results"]
    }


def test_multiturn_matching_request_applies_limit_and_hard_exclusions(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    initial = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "message": (
                "I have garlic, onion and bread. Suggest five recipes "
                "without peanuts."
            )
        },
    ).json()

    assert initial["state"]["available_ingredients"] == [
        "garlic",
        "onion",
        "bread",
    ]
    assert initial["state"]["hard_exclusions"] == ["peanuts"]
    initial_report = initial["result"].get("report", initial["result"])
    assert len(initial_report["results"]) <= 5
    assert "peanut-garlic" not in {
        item["recipe_id"] for item in initial_report["results"]
    }

    preferred = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Prefer fewer additional ingredients."},
    ).json()
    assert preferred["state"]["hard_exclusions"] == ["peanuts"]
    assert preferred["state"]["ranking_preferences"] == ["fewer_additional"]
    preferred_report = preferred["result"].get("report", preferred["result"])
    assert [item["recipe_id"] for item in preferred_report["results"]] != [
        item["recipe_id"] for item in initial_report["results"]
    ]

    restricted = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Exclude egg too."},
    ).json()
    assert set(restricted["state"]["hard_exclusions"]) == {"peanuts", "egg"}
    restricted_report = restricted["result"].get("report", restricted["result"])
    assert "garlic-bread-egg" not in {
        item["recipe_id"] for item in restricted_report["results"]
    }


def test_fallback_parses_assignment_prompts_without_added_exclusions(
) -> None:
    state = new_session_state("parser-test")
    search = _parse_fallback_request("Find a penne pasta recipe.", state)
    stats = _parse_fallback_request(
        "How many recipes contain both garlic and onion?", state
    )
    ranking = _parse_fallback_request(
        "Among recipes containing tomato, show five with the fewest instruction steps.",
        state,
    )
    matching = _parse_fallback_request(
        "I have eggs, onions, tomatoes and rice. Suggest five recipes without "
        "peanuts and with at most six instruction steps.",
        state,
    )

    assert search is not None and search.search_query == "penne pasta"
    assert stats is not None
    assert stats.statistic == "ingredient_prevalence"
    assert stats.statistic_ingredient == "garlic onion"
    assert ranking is not None
    assert ranking.available_ingredients == ("tomato",)
    assert ranking.ranking_preferences == ("fewer_steps",)
    assert ranking.limit == 5
    assert matching is not None
    assert matching.exclude_ingredients == ("peanuts",)
    assert matching.max_instruction_steps == 6
    assert matching.limit == 5


def test_statistics_conversation_answers_count_and_percentage(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    count = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "How many recipes contain both garlic and onion?"},
    ).json()
    percentage = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "What percentage of the dataset is that?"},
    ).json()

    statistic = count["result"]["statistics"][0]
    percentage_statistic = percentage["result"]["statistics"][0]
    assert statistic["statistic"] == "ingredient_prevalence"
    assert statistic["numerator"] == 5
    assert statistic["denominator"] == 10
    assert "Search for 'all recipes'" not in count["response"]
    assert f"percentage of analyzed recipes {statistic['value'] * 100:.6f}%" in percentage["response"]
    assert percentage_statistic["numerator"] == statistic["numerator"]
    assert percentage_statistic["denominator"] == statistic["denominator"]


def test_conversational_search_includes_ingredient_only_matches(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    response = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Find butter and show the details."},
    ).json()

    assert response["status"] == "complete"
    assert response["result"]["recipes"]
    assert any(
        "butter" in " ".join(
            recipe["original_record"].get("ingredients", [])
        ).casefold()
        for recipe in response["result"]["recipes"]
    )


def test_search_dairy_followup_refines_filters_without_losing_context(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Find garlic bread"},
    )
    refined = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "I cannot have dairy."},
    ).json()

    assert refined["status"] == "complete"
    assert "cream" in refined["state"]["filters"]["exclude_ingredients"]
    assert "butter" in refined["state"]["filters"]["exclude_ingredients"]
    assert all(
        result_id.split(":", 1)[1] not in {
            "garlic-bread-cream",
            "garlic-bread-egg",
            "garlic-bread-cream",
        }
        for result_id in refined["state"]["last_result_ids"]
    )


def test_matching_followups_preserve_requested_result_limit(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    initial = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={
            "message": (
                "I have garlic, onion and bread. Suggest five recipes "
                "without peanuts."
            )
        },
    ).json()
    followup = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Prefer fewer additional ingredients."},
    ).json()

    initial_report = initial["result"]["report"]
    assert initial["state"]["result_limit"] == 5
    assert len(initial_report["results"]) <= 5
    assert len(followup["result"]["results"]) <= 5


def test_concurrent_session_followups_are_serialized(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Find garlic bread"},
    )

    def send(message: str) -> int:
        return client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": message},
        ).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(send, ("Exclude egg too.", "Exclude peanut too.")))
    state = client.get(f"/api/v1/sessions/{session_id}").json()["state"]

    assert statuses == [200, 200]
    assert set(state["filters"]["exclude_ingredients"]) == {"egg", "peanut"}


def test_gemini_disabled_and_invalid_requests_fall_back_or_reject_cleanly(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    natural = client.post(
        "/api/v1/recipes/search",
        json={"natural_language": "Find garlic bread"},
    )
    invalid = client.post(
        "/api/v1/matching",
        json={"available_ingredients": [], "unexpected": True},
    )
    too_large_message = client.post(
        "/api/v1/sessions",
    )

    assert natural.status_code == 200
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_request"
    assert too_large_message.status_code == 201


def test_request_size_limit_and_gemini_failure_fallback(
    api: tuple[TestClient, Path],
) -> None:
    client, database_path = api
    large = client.post(
        "/api/v1/recipes/search",
        content=b" " * (64 * 1024 + 1),
        headers={"content-type": "application/json"},
    )
    assert large.status_code == 413

    from recipe_intelligence.llm import LLMTimeout

    class FailingProvider:
        def generate_json(self, **kwargs: object) -> dict[str, object]:
            _ = kwargs
            raise LLMTimeout("test timeout")

    app = create_app(database_path, provider=FailingProvider())
    with TestClient(app) as failed_client:
        session_id = failed_client.post("/api/v1/sessions").json()["session_id"]
        response = failed_client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "Find garlic bread and show details"},
        )
    assert response.status_code == 200
    assert response.json()["used_gemini"] is False
    assert response.json()["result"]["recipes"]
    assert response.json()["result"]["recipes"][0]["original_record"]["instructions"]


def test_substitution_unknown_recipe_and_ambiguous_id_are_handled(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    missing = client.post(
        "/api/v1/substitutions",
        json={
            "recipe_id": "missing",
            "source": "ar",
            "ingredient_to_replace": "cream",
        },
    )
    assert missing.status_code == 404


def test_failed_followup_preserves_state(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    initial = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Find garlic bread"},
    ).json()["state"]
    failed = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Replace cream in that recipe."},
    ).json()
    persisted = client.get(f"/api/v1/sessions/{session_id}").json()["state"]

    assert failed["status"] == "clarification"
    assert persisted == initial


def test_invalid_substitution_followup_does_not_update_session_state(
    api: tuple[TestClient, Path],
) -> None:
    client, _ = api
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Find creamy garlic bread"},
    )
    client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Replace cream in that recipe."},
    )
    before = client.get(f"/api/v1/sessions/{session_id}").json()["state"]

    failed = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "Replace anchovy in that recipe."},
    )
    after = client.get(f"/api/v1/sessions/{session_id}").json()["state"]

    assert failed.status_code == 422
    assert after == before
