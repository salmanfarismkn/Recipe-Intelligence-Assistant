"""Evaluate grounded assistant behavior against a local source archive."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from recipe_intelligence.assistant import (
    execute_request,
    generate_grounded_response,
    parse_request_interpretation,
)
from recipe_intelligence.database import connect_database, ingest_dataset_archive


class SelectAllFacts:
    def generate_json(
        self,
        *,
        system_instruction: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        _ = system_instruction, schema
        return {
            "opening": "results",
            "fact_ids": [fact["fact_id"] for fact in payload["facts"]],
        }


class SelectInventedFact:
    def generate_json(
        self,
        *,
        system_instruction: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        _ = system_instruction, payload, schema
        return {"opening": "results", "fact_ids": ["invented-recipe-id"]}


def _request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
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
        "limit": 3,
    }
    request.update(overrides)
    return request


def _evaluate(connection: sqlite3.Connection) -> dict[str, Any]:
    interpretation = parse_request_interpretation(
        _request(
            task="recipe_search",
            search_query="garlic bread",
            dish_name="garlic bread",
            search_field="all",
            details_requested=True,
        ),
        "Find garlic bread recipes and show details",
    )
    if interpretation.request is None:
        raise RuntimeError("The fixed evaluation request unexpectedly needs clarification")
    search = execute_request(connection, interpretation.request)
    selected = generate_grounded_response(search, SelectAllFacts())
    invalid_selection = generate_grounded_response(search, SelectInventedFact())
    search_ids = [
        f"{recipe['source']}:{recipe['recipe_id']}"
        for recipe in search.payload["recipes"]
    ]
    references_verified = all(
        reference in selected.text for reference in search_ids
    )
    invented_id_rejected = (
        not invalid_selection.used_gemini
        and "invented-recipe-id" not in invalid_selection.text
    )

    matching_resolution = parse_request_interpretation(
        _request(
            task="ingredient_matching",
            available_ingredients=["garlic"],
            exclude_ingredients=["peanut"],
        ),
        "I have garlic; avoid peanut",
    )
    if matching_resolution.request is None:
        raise RuntimeError("The fixed matching request unexpectedly needs clarification")
    matching = execute_request(connection, matching_resolution.request)
    report = matching.payload["report"]
    matching_ids = [
        f"{recipe['source']}:{recipe['recipe_id']}"
        for recipe in report["results"]
    ]
    exclusion_preserved = all(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM recipe_ingredients
            WHERE source = ? AND recipe_id = ?
              AND canonical_name LIKE '%peanut%'
            """,
            (recipe["source"], recipe["recipe_id"]),
        ).fetchone()[0]
        == 0
        for recipe in report["results"]
    )

    checks = {
        "interpretation_entity_validation": bool(search_ids),
        "search_references_preserved": references_verified,
        "fabricated_reference_rejected": invented_id_rejected,
        "hard_exclusion_preserved": exclusion_preserved,
    }
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "search_recipe_ids": search_ids,
        "matching_candidates_analyzed": report["candidate_recipes_analyzed"],
        "matching_recipe_ids": matching_ids,
        "matching_exclusion_preserved": exclusion_preserved,
    }


def evaluate_archive(archive_path: Path) -> dict[str, Any]:
    connection = connect_database(":memory:")
    try:
        inventory = ingest_dataset_archive(connection, archive_path)
        results = _evaluate(connection)
        return {
            "archive_matches_pinned_snapshot": inventory.archive_matches_pinned_snapshot,
            "total_record_count": inventory.total_record_count,
            "source_record_counts": {
                source.source: source.record_count for source in inventory.sources
            },
            **results,
        }
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate grounded assistant behavior against a source archive."
    )
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    results = evaluate_archive(args.archive)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
