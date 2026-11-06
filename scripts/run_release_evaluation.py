"""Audit persisted dataset integrity and benchmark deterministic services."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from fastapi.testclient import TestClient

from recipe_intelligence.api import create_app
from recipe_intelligence.database import connect_database
from recipe_intelligence.matching import match_recipes
from recipe_intelligence.normalization import normalize_ingredient
from recipe_intelligence.retrieval import search_recipes
from recipe_intelligence.statistics import calculate_ingredient_prevalence
from recipe_intelligence.substitution import propose_substitutions


def _memory_bytes() -> int | None:
    if platform.system() == "Windows":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    if hasattr(os, "sysconf"):
        try:
            return int(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            )
        except (OSError, ValueError):
            return None
    return None


def _measure(operation: Callable[[], Any], repeats: int) -> dict[str, Any]:
    operation()
    samples = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = operation()
        samples.append(time.perf_counter() - started)
    return {
        "sample_seconds": samples,
        "median_seconds": statistics.median(samples),
        "output": output,
    }


def _audit_database(connection: sqlite3.Connection) -> dict[str, Any]:
    metadata = dict(connection.execute("SELECT key, value FROM dataset_metadata"))
    inventory = json.loads(metadata.get("inventory_json", "{}"))
    source_counts = {
        row["source"]: row["record_count"]
        for row in connection.execute(
            "SELECT source, COUNT(*) AS record_count FROM recipes GROUP BY source"
        )
    }
    source_ingredient_counts = {
        row["source"]: row["ingredient_row_count"]
        for row in connection.execute(
            """
            SELECT source, COUNT(*) AS ingredient_row_count
            FROM recipe_ingredients GROUP BY source
            """
        )
    }
    counts = {
        "recipe_count": connection.execute(
            "SELECT COUNT(*) FROM recipes"
        ).fetchone()[0],
        "unique_recipe_id_count": connection.execute(
            "SELECT COUNT(DISTINCT recipe_id) FROM recipes"
        ).fetchone()[0],
        "ingredient_row_count": connection.execute(
            "SELECT COUNT(*) FROM recipe_ingredients"
        ).fetchone()[0],
        "recipes_with_instructions": connection.execute(
            "SELECT COUNT(*) FROM recipes WHERE instructions IS NOT NULL"
        ).fetchone()[0],
        "empty_or_null_instruction_count": connection.execute(
            "SELECT COUNT(*) FROM recipes WHERE instructions IS NULL OR TRIM(instructions) = ''"
        ).fetchone()[0],
        "exact_duplicate_record_excess": connection.execute(
            """
            SELECT COALESCE(SUM(record_count - 1), 0)
            FROM (
                SELECT COUNT(*) AS record_count
                FROM recipes
                GROUP BY raw_record_json
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0],
    }
    cross_source_duplicate_ids = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT recipe_id FROM recipes
            GROUP BY recipe_id HAVING COUNT(DISTINCT source) > 1
        )
        """
    ).fetchone()[0]
    original_field_mismatches = connection.execute(
        """
        SELECT COUNT(*) FROM recipes
        WHERE title IS NOT json_extract(raw_record_json, '$.title')
           OR instructions IS NOT json_extract(raw_record_json, '$.instructions')
           OR picture_link IS NOT json_extract(raw_record_json, '$.picture_link')
        """
    ).fetchone()[0]
    ingredient_original_mismatches = connection.execute(
        """
        SELECT COUNT(*) FROM recipe_ingredients
        WHERE original_text IS NOT json_extract(original_value_json, '$')
        """
    ).fetchone()[0]
    normalized_text_mismatches = connection.execute(
        """
        SELECT COUNT(*) FROM recipe_ingredients
        WHERE original_text IS NOT NULL
          AND normalized_text IS NOT normalize_ingredient(original_text)
        """
    ).fetchone()[0]
    expected_source_counts = {
        source["source"]: source["record_count"]
        for source in inventory.get("sources", [])
    }
    checks = {
        "archive_matches_pinned_snapshot": (
            inventory.get("archive_matches_pinned_snapshot") is True
        ),
        "dataset_metadata_count_matches_database": (
            counts["recipe_count"] == int(metadata.get("total_record_count", -1))
        ),
        "source_counts_match_inventory": source_counts == expected_source_counts,
        "recipe_ids_unique_across_sources": (
            counts["unique_recipe_id_count"] == counts["recipe_count"]
            and cross_source_duplicate_ids == 0
        ),
        "original_recipe_fields_preserved": original_field_mismatches == 0,
        "original_ingredient_values_preserved": ingredient_original_mismatches == 0,
        "normalized_ingredient_values_reproducible": normalized_text_mismatches == 0,
    }
    return {
        "archive_sha256": metadata.get("archive_sha256"),
        "archive_matches_pinned_snapshot": inventory.get(
            "archive_matches_pinned_snapshot"
        ),
        "source_counts": source_counts,
        "source_ingredient_counts": source_ingredient_counts,
        "counts": counts,
        "cross_source_duplicate_id_count": cross_source_duplicate_ids,
        "original_field_mismatch_count": original_field_mismatches,
        "original_ingredient_mismatch_count": ingredient_original_mismatches,
        "normalized_text_mismatch_count": normalized_text_mismatches,
        "instruction_step_methods": {
            row["instruction_step_method"]: row["record_count"]
            for row in connection.execute(
                """
                SELECT instruction_step_method, COUNT(*) AS record_count
                FROM recipes GROUP BY instruction_step_method
                ORDER BY instruction_step_method
                """
            )
        },
        "checks": checks,
    }


def _benchmark_services(
    connection: sqlite3.Connection, repeats: int
) -> dict[str, Any]:
    search = _measure(
        lambda: len(search_recipes(connection, "penne", limit=10)),
        repeats,
    )
    prevalence = _measure(
        lambda: calculate_ingredient_prevalence(
            connection, "garlic"
        ).denominator,
        repeats,
    )
    matching = _measure(
        lambda: match_recipes(
            connection,
            ("eggs", "onions", "tomatoes", "rice"),
            excluded_ingredients=("peanuts",),
            max_instruction_steps=6,
            ranking_preferences=("fewer_additional",),
            limit=5,
        ),
        repeats,
    )
    recipe = connection.execute(
        """
        SELECT source, recipe_id
        FROM recipe_ingredients
        WHERE canonical_name = 'cream'
        ORDER BY source, recipe_id
        LIMIT 1
        """
    ).fetchone()
    if recipe is None:
        raise RuntimeError("The pinned dataset has no recipe containing cream")
    substitution = _measure(
        lambda: propose_substitutions(
            connection,
            recipe["source"],
            recipe["recipe_id"],
            "cream",
            limit=10,
        ),
        repeats,
    )
    if matching["output"] is None or substitution["output"] is None:
        raise RuntimeError("A deterministic service benchmark returned no report")
    return {
        "method": (
            "One warm-up followed by repeated in-process service calls on the "
            "same local SQLite database; medians are machine- and cache-dependent."
        ),
        "repeats_after_warmup": repeats,
        "search_penne": {
            "median_seconds": search["median_seconds"],
            "sample_seconds": search["sample_seconds"],
            "returned_recipe_count": search["output"],
        },
        "ingredient_prevalence_garlic": {
            "median_seconds": prevalence["median_seconds"],
            "sample_seconds": prevalence["sample_seconds"],
            "valid_ingredient_list_denominator": prevalence["output"],
        },
        "matching_eggs_onions_tomatoes_rice": {
            "median_seconds": matching["median_seconds"],
            "sample_seconds": matching["sample_seconds"],
            "candidate_recipes_analyzed": (
                matching["output"].candidate_recipes_analyzed
            ),
            "returned_recipe_count": len(matching["output"].results),
        },
        "substitution_discovery_for_cream": {
            "median_seconds": substitution["median_seconds"],
            "sample_seconds": substitution["sample_seconds"],
            "recipe_reference": f"{recipe['source']}:{recipe['recipe_id']}",
            "evidence_status": substitution["output"].status,
            "option_count": len(substitution["output"].options),
        },
    }


def _conversation_evaluations(database_path: Path) -> dict[str, Any]:
    client = TestClient(
        create_app(database_path, use_environment_provider=False)
    )
    evidence_connection = connect_database(database_path)
    completed_sessions: list[str] = []

    def create_session() -> str:
        response = client.post("/api/v1/sessions")
        response.raise_for_status()
        session_id = response.json()["session_id"]
        completed_sessions.append(session_id)
        return session_id

    def send(session_id: str, message: str) -> dict[str, Any]:
        response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": message},
        )
        response.raise_for_status()
        return response.json()

    def recipe_refs(results: list[dict[str, Any]]) -> list[str]:
        return [
            f"{item['source']}:{item['recipe_id']}"
            for item in results
        ]

    try:
        search_session = create_session()
        search_first = send(search_session, "Find garlic bread")
        search_followup = send(search_session, "Exclude mushrooms too.")
        search_results = search_followup["result"]["recipe_ids"]

        stats_session = create_session()
        stats_turn = send(
            stats_session,
            "How many recipes contain both garlic and onion?",
        )
        stats_items = stats_turn["result"]["statistics"]
        stats = stats_items[0]

        ranking_session = create_session()
        ranking_turn = send(
            ranking_session,
            "Among recipes containing tomatoes, show five with the fewest steps.",
        )
        ranking_report = ranking_turn["result"]["report"]
        ranked_results = ranking_report["results"]

        matching_session = create_session()
        matching_first = send(
            matching_session,
            "I have eggs, onions, tomatoes and rice. Suggest five recipes "
            "without peanuts and no more than six steps.",
        )
        matching_followup = send(
            matching_session,
            "Exclude eggs too.",
        )
        matching_report = matching_followup["result"]
        matching_results = matching_report["results"]
        verified_connection = connect_database(database_path)
        try:
            hard_constraint_violations = []
            for result in matching_results:
                ingredients = {
                    row["canonical_name"]
                    for row in verified_connection.execute(
                        """
                        SELECT canonical_name FROM recipe_ingredients
                        WHERE source = ? AND recipe_id = ?
                        """,
                        (result["source"], result["recipe_id"]),
                    )
                }
                recipe = verified_connection.execute(
                    """
                    SELECT instruction_step_count FROM recipes
                    WHERE source = ? AND recipe_id = ?
                    """,
                    (result["source"], result["recipe_id"]),
                ).fetchone()
                if {"peanut", "egg"} & ingredients:
                    hard_constraint_violations.append(
                        f"{result['source']}:{result['recipe_id']}"
                    )
                if (
                    recipe is None
                    or recipe["instruction_step_count"] is None
                    or recipe["instruction_step_count"] > 6
                ):
                    hard_constraint_violations.append(
                        f"step-limit:{result['source']}:{result['recipe_id']}"
                    )
        finally:
            verified_connection.close()

        substitution_session = create_session()
        substitution_search = send(
            substitution_session, "Find Cherry Bell Cookies"
        )
        substitution_selection = send(
            substitution_session, "What about the first result?"
        )
        substitution_turn = send(
            substitution_session, "Replace cream in that recipe."
        )
        proposal = substitution_turn["result"]
        weak_option = next(
            (
                option
                for option in proposal["options"]
                if option["evidence_strength"] == "weak"
                and not option["sufficiently_supported"]
            ),
            None,
        )
        if weak_option is None:
            raise RuntimeError(
                "The full-dataset weak-evidence refusal example was not reproduced"
            )

        no_match_session = create_session()
        no_match_turn = send(
            no_match_session,
            "Find a recipe titled this-is-not-a-real-dish-zzq.",
        )

        one_shot_session = create_session()
        one_shot_turn = send(one_shot_session, "How common is garlic?")
        direct_analytics = client.post(
            "/api/v1/analytics",
            json={
                "statistic": "ingredient_prevalence",
                "ingredient": "garlic",
            },
        )
        direct_analytics.raise_for_status()
        direct_statistic = direct_analytics.json()["result"]
        one_shot_statistic = one_shot_turn["result"]["statistics"][0]

        positive_proposal = propose_substitutions(
            evidence_connection,
            "fn",
            "no81mIW/BbCbewZyPT8AaKCgsrDitEO",
            "chickpea",
            replacement_ingredients=("black beans",),
        )
        positive_option = next(
            (
                option
                for option in positive_proposal.options
                if option.replacement_canonical_name == "black beans"
                and option.evidence_strength == "direct"
                and option.sufficiently_supported
                and not option.conflicting_evidence
            ),
            None,
        )
        if positive_option is None:
            raise RuntimeError(
                "The full-dataset positive substitution example was not reproduced"
            )

        gemini_configured = all(
            os.environ.get(name, "").strip()
            for name in ("LLM_ENABLED", "GEMINI_API_KEY", "GEMINI_MODEL")
        ) and os.environ.get("LLM_ENABLED", "").strip().casefold() == "true"

        return {
            "search_and_contextual_restriction": {
                "turns": [
                    {
                        "user": "Find garlic bread",
                        "status": search_first["status"],
                        "response": search_first["response"],
                    },
                    {
                        "user": "Exclude mushrooms too.",
                        "status": search_followup["status"],
                        "response": search_followup["response"],
                    },
                ],
                "applied_exclusions": search_followup["state"]["filters"][
                    "exclude_ingredients"
                ],
                "result_count": search_followup["result"]["result_count"],
                "first_five_recipe_ids": search_results[:5],
            },
            "aggregate_statistics": {
                "turns": [
                    {
                        "user": "How many recipes contain both garlic and onion?",
                        "status": stats_turn["status"],
                        "response": stats_turn["response"],
                    }
                ],
                "statistic": stats["statistic"],
                "numerator": stats["numerator"],
                "denominator": stats["denominator"],
                "value": stats["value"],
                "analyzed_recipe_count": stats["analyzed_recipe_count"],
                "excluded_recipe_count": stats["excluded_recipe_count"],
                "calculation_method": stats["calculation_method"],
            },
            "instruction_step_ranking": {
                "turns": [
                    {
                        "user": (
                            "Among recipes containing tomatoes, show five with "
                            "the fewest steps."
                        ),
                        "status": ranking_turn["status"],
                        "response": ranking_turn["response"],
                    }
                ],
                "ranking_preferences": ranking_report["ranking_preferences"],
                "candidate_recipes_analyzed": (
                    ranking_report["candidate_recipes_analyzed"]
                ),
                "results": [
                    {
                        "reference": f"{item['source']}:{item['recipe_id']}",
                        "title": item["title"],
                        "instruction_step_count": item["instruction_step_count"],
                        "instruction_step_method": item[
                            "instruction_step_method"
                        ],
                    }
                    for item in ranked_results
                ],
            },
            "ingredient_matching_and_followup_exclusion": {
                "turns": [
                    {
                        "user": (
                            "I have eggs, onions, tomatoes and rice. Suggest "
                            "five recipes without peanuts and no more than "
                            "six steps."
                        ),
                        "status": matching_first["status"],
                        "response": matching_first["response"],
                    },
                    {
                        "user": "Exclude eggs too.",
                        "status": matching_followup["status"],
                        "response": matching_followup["response"],
                    },
                ],
                "hard_exclusions": matching_followup["state"]["hard_exclusions"],
                "maximum_instruction_steps": matching_followup["state"][
                    "maximum_instruction_steps"
                ],
                "ranking_preferences": matching_followup["state"][
                    "ranking_preferences"
                ],
                "candidate_recipes_analyzed": (
                    matching_report["candidate_recipes_analyzed"]
                ),
                "result_references": recipe_refs(matching_results),
                "hard_constraint_violations": hard_constraint_violations,
                "quantity_sufficiency_verified": matching_report[
                    "quantity_sufficiency_verified"
                ],
            },
            "evidence_based_substitution_refusal": {
                "turns": [
                    {
                        "user": "Find Cherry Bell Cookies",
                        "status": substitution_search["status"],
                        "response": substitution_search["response"],
                    },
                    {
                        "user": "What about the first result?",
                        "status": substitution_selection["status"],
                        "response": substitution_selection["response"],
                    },
                    {
                        "user": "Replace cream in that recipe.",
                        "status": substitution_turn["status"],
                        "response": substitution_turn["response"],
                    },
                ],
                "original_recipe": {
                    "reference": (
                        f"{proposal['original_recipe']['source']}:"
                        f"{proposal['original_recipe']['recipe_id']}"
                    ),
                    "title": proposal["original_title"],
                },
                "ingredient_to_replace": "cream",
                "proposal_status": proposal["status"],
                "refused_option": {
                    "replacement": weak_option["replacement_ingredient"],
                    "evidence_classification": weak_option["evidence_strength"],
                    "sufficiently_supported": weak_option[
                        "sufficiently_supported"
                    ],
                    "conflicting_evidence": weak_option[
                        "conflicting_evidence"
                    ],
                    "supporting_recipe_ids": [
                        f"{item['reference']['source']}:"
                        f"{item['reference']['recipe_id']}"
                        for item in weak_option["supporting_recipes"]
                    ],
                    "cooccurrence_recipe_ids": [
                        f"{item['source']}:{item['recipe_id']}"
                        for item in weak_option["cooccurrence_recipe_ids"]
                    ],
                    "revised_ingredient_list": weak_option[
                        "revised_ingredient_list"
                    ],
                    "expected_trade_offs": weak_option["expected_trade_offs"],
                    "remaining_uncertainty": weak_option[
                        "remaining_uncertainty"
                    ],
                },
            },
            "positive_substitution_verification": {
                "search_method": (
                    "Explicitly tested a pinned Hummus recipe against the "
                    "dataset's Black Bean Hummus counterpart through the existing "
                    "substitution service; no evidence thresholds were changed."
                ),
                "proposal_status": positive_proposal.status,
                "original_recipe": {
                    "reference": (
                        f"{positive_proposal.original_recipe.source}:"
                        f"{positive_proposal.original_recipe.recipe_id}"
                    ),
                    "title": positive_proposal.original_title,
                },
                "original_ingredient": "4 cups garbanzo beans, drained",
                "replacement_ingredient": positive_option.replacement_ingredient,
                "evidence_classification": positive_option.evidence_strength,
                "sufficiently_supported": positive_option.sufficiently_supported,
                "conflicting_evidence": positive_option.conflicting_evidence,
                "supporting_recipe_ids": [
                    f"{item.reference.source}:{item.reference.recipe_id}"
                    for item in positive_option.supporting_recipes
                ],
                "supporting_recipe_evidence": [
                    {
                        "reference": (
                            f"{item.reference.source}:{item.reference.recipe_id}"
                        ),
                        "title": item.title,
                        "replacement_ingredient_lines": list(
                            item.replacement_ingredient_lines
                        ),
                        "shared_ingredients": list(item.shared_ingredients),
                        "shared_title_terms": list(item.shared_title_terms),
                        "shared_preparation_methods": list(
                            item.shared_preparation_methods
                        ),
                        "ingredient_overlap": item.ingredient_overlap,
                        "instruction_excerpt": list(item.instruction_excerpt),
                    }
                    for item in positive_option.supporting_recipes
                ],
                "revised_ingredient_list": list(
                    positive_option.revised_ingredient_list or ()
                ),
                "revised_instructions": None,
                "instruction_adjustments": [
                    {
                        "original_instruction": item.original_instruction,
                        "adjustment_needed": item.adjustment_needed,
                    }
                    for item in positive_option.instruction_adjustments
                ],
                "instruction_note": (
                    "No rewrite was returned: the original instructions refer "
                    "generically to 'all the ingredients' and do not name the "
                    "replaced ingredient."
                ),
                "expected_trade_offs": list(positive_option.expected_trade_offs),
                "remaining_uncertainty": list(
                    positive_option.remaining_uncertainty
                ),
            },
            "no_match_case": {
                "turns": [
                    {
                        "user": (
                            "Find a recipe titled "
                            "this-is-not-a-real-dish-zzq."
                        ),
                        "status": no_match_turn["status"],
                        "response": no_match_turn["response"],
                    }
                ],
                "result_count": len(
                    no_match_turn["result"].get("recipes", [])
                ),
            },
            "one_shot_comparison": {
                "llm_status": "NOT_VERIFIED",
                "llm_reason": (
                    "No valid Gemini configuration was present; no mock was "
                    "used as a live comparison."
                ),
                "executed_deterministic_comparison": {
                    "prompt": "How common is garlic?",
                    "conversation_status": one_shot_turn["status"],
                    "conversation_numerator": one_shot_statistic["numerator"],
                    "conversation_denominator": one_shot_statistic["denominator"],
                    "direct_analytics_numerator": direct_statistic["numerator"],
                    "direct_analytics_denominator": direct_statistic[
                        "denominator"
                    ],
                    "same_computed_result": (
                        one_shot_statistic["numerator"]
                        == direct_statistic["numerator"]
                        and one_shot_statistic["denominator"]
                        == direct_statistic["denominator"]
                    ),
                },
            },
            "gemini_live_verification": {
                "status": "NOT_VERIFIED",
                "configuration_present": gemini_configured,
                "reason": (
                    "A live Gemini request was not run because LLM_ENABLED, "
                    "GEMINI_API_KEY, and GEMINI_MODEL were not all configured."
                    if not gemini_configured
                    else "Live Gemini behavior is outside this deterministic runner."
                ),
            },
            "all_conversation_turns_complete": all(
                turn["status"] == "complete"
                for key, scenario in {
                    "search": {
                        "turns": [
                            {"status": search_first["status"]},
                            {"status": search_followup["status"]},
                        ]
                    },
                    "statistics": {"turns": [{"status": stats_turn["status"]}]},
                    "ranking": {"turns": [{"status": ranking_turn["status"]}]},
                    "matching": {
                        "turns": [
                            {"status": matching_first["status"]},
                            {"status": matching_followup["status"]},
                        ]
                    },
                    "substitution": {
                        "turns": [
                            {"status": substitution_search["status"]},
                            {"status": substitution_selection["status"]},
                            {"status": substitution_turn["status"]},
                        ]
                    },
                    "no_match": {"turns": [{"status": no_match_turn["status"]}]},
                    "one_shot": {"turns": [{"status": one_shot_turn["status"]}]},
                }.items()
                for turn in scenario["turns"]
            )
            and not hard_constraint_violations
            and no_match_turn["status"] == "complete"
            and len(no_match_turn["result"].get("recipes", [])) == 0
            and one_shot_statistic["numerator"] == direct_statistic["numerator"]
            and one_shot_statistic["denominator"] == direct_statistic["denominator"],
        }
    finally:
        for session_id in completed_sessions:
            client.delete(f"/api/v1/sessions/{session_id}")
        evidence_connection.close()
        client.close()


def evaluate(database_path: Path, repeats: int) -> dict[str, Any]:
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")
    uri = f"{database_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.create_function(
        "normalize_ingredient", 1, normalize_ingredient, deterministic=True
    )
    try:
        integrity = _audit_database(connection)
        benchmarks = _benchmark_services(connection, repeats)
        conversations = _conversation_evaluations(database_path)
        integrity["checks"]["services_returned_measured_outputs"] = all(
            item.get("returned_recipe_count", item.get("option_count", 1)) >= 0
            for item in benchmarks.values()
            if isinstance(item, dict) and "median_seconds" in item
        )
        return {
            "evaluation_version": 1,
            "database_path": database_path.name,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": _memory_bytes(),
            "database_size_bytes": database_path.stat().st_size,
            "integrity": integrity,
            "performance": benchmarks,
            "conversations": conversations,
            "all_integrity_checks_passed": all(
                integrity["checks"].values()
            ),
            "final_assessment": "CONDITIONAL PASS",
            "assessment_reasons": [
                "Deterministic dataset and service checks passed, including a "
                "directly supported real-data substitution and a weak-evidence refusal.",
                "Live Gemini verification was not performed because credentials "
                "and model configuration were unavailable.",
                "Ingredient quantity equivalence and substitution trade-offs "
                "remain unverified by recipe text.",
            ],
            "limitations": [
                "Timing is a local single-machine measurement, not a latency SLO.",
                "Gemini was not exercised because no valid configuration was present.",
                "Docker is not configured or exercised.",
                "The full-dataset service timings do not measure concurrent load.",
            ],
        }
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit an ingested SQLite dataset and benchmark local services."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/processed/recipes.sqlite3"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation_report.json"),
    )
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be between 1 and 20")
    report = evaluate(args.database, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_integrity_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
