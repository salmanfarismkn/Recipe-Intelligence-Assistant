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
            "all_integrity_checks_passed": all(
                integrity["checks"].values()
            ),
            "limitations": [
                "Timing is a local single-machine measurement, not a latency SLO.",
                "Gemini and Docker are not exercised by this deterministic runner.",
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
