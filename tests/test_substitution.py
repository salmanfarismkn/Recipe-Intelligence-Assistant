from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from recipe_intelligence.database import connect_database, ingest_dataset_archive
from recipe_intelligence.ingestion import DATASET_FILES
from recipe_intelligence.matching import parse_ingredient
from recipe_intelligence.substitution import propose_substitutions


@pytest.fixture
def connection() -> sqlite3.Connection:
    database = connect_database(":memory:")
    yield database
    database.close()


def fixture_records() -> dict[str, dict[str, object]]:
    return {
        "apple-muffin-source": {
            "title": "Apple Breakfast Muffins",
            "ingredients": [
                "2 large eggs",
                "2 cups flour",
                "1 cup apple",
                "1/2 cup sugar",
                "1 teaspoon vanilla",
            ],
            "instructions": (
                "1. Mix eggs, flour, apple, sugar, and vanilla.\n"
                "2. Bake until set."
            ),
        },
        "apple-muffin-flax": {
            "title": "Apple Breakfast Muffins with Ground Flaxseed",
            "ingredients": [
                "1 tablespoon ground flaxseed",
                "2 cups flour",
                "1 cup apple",
                "1/2 cup sugar",
                "1 teaspoon vanilla",
            ],
            "instructions": (
                "1. Mix ground flaxseed, flour, apple, sugar, and vanilla.\n"
                "2. Bake until set."
            ),
        },
        "apple-muffin-flax-second": {
            "title": "Apple Breakfast Muffins with Flaxseed",
            "ingredients": [
                "1 tablespoon ground flaxseed",
                "2 cups flour",
                "1 cup apple",
                "1/2 cup sugar",
                "1 teaspoon vanilla",
            ],
            "instructions": (
                "1. Mix flaxseed, flour, apple, sugar, and vanilla.\n"
                "2. Bake until set."
            ),
        },
        "apple-muffin-banana": {
            "title": "Apple Breakfast Muffins with Banana",
            "ingredients": [
                "1 banana",
                "2 cups flour",
                "1 cup apple",
                "1/2 cup sugar",
                "1 teaspoon vanilla",
            ],
            "instructions": (
                "1. Mix banana, flour, apple, sugar, and vanilla.\n"
                "2. Bake until set."
            ),
        },
        "apple-muffin-honey": {
            "title": "Apple Breakfast Muffins with Honey",
            "ingredients": [
                "honey",
                "2 cups flour",
                "1 cup apple",
                "1 cup oats",
            ],
            "instructions": (
                "1. Mix honey, flour, apple, and oats.\n"
                "2. Bake until set."
            ),
        },
        "egg-flax-cooccurrence": {
            "title": "Apple Muffins with Egg and Flaxseed",
            "ingredients": [
                "2 eggs",
                "1 tablespoon ground flaxseed",
                "2 cups flour",
                "1 cup apple",
                "1/2 cup sugar",
                "1 teaspoon vanilla",
            ],
            "instructions": "Bake.",
        },
        "egg-banana-cooccurrence": {
            "title": "Apple Muffins with Egg and Banana",
            "ingredients": [
                "2 eggs",
                "1 banana",
                "2 cups flour",
                "1 cup apple",
                "1/2 cup sugar",
                "1 teaspoon vanilla",
            ],
            "instructions": "Mix and bake.",
        },
        "egg-honey-cooccurrence": {
            "title": "Apple Muffins with Egg and Honey",
            "ingredients": [
                "2 eggs",
                "honey",
                "2 cups flour",
                "1 cup apple",
                "1/2 cup sugar",
            ],
            "instructions": "Mix and bake.",
        },
        "turmeric-soup": {
            "title": "Turmeric Lentil Soup",
            "ingredients": ["turmeric", "lentils", "vegetable broth"],
            "instructions": "Simmer until tender.",
        },
        "savory-soup-source": {
            "title": "Chicken and Vegetable Soup",
            "ingredients": [
                "1 pound chicken breast",
                "1 onion",
                "2 cloves garlic",
                "4 cups vegetable broth",
                "2 carrots",
                "1 teaspoon thyme",
            ],
            "instructions": (
                "1. Simmer chicken, onion, garlic, carrots, and thyme in broth.\n"
                "2. Serve hot."
            ),
        },
        "savory-soup-tofu": {
            "title": "Savory Tofu and Vegetable Soup",
            "ingredients": [
                "1 block tofu",
                "1 onion",
                "2 cloves garlic",
                "4 cups vegetable broth",
                "2 carrots",
                "1 teaspoon thyme",
            ],
            "instructions": (
                "1. Simmer tofu, onion, garlic, carrots, and thyme in broth.\n"
                "2. Serve hot."
            ),
        },
        "milk-cake-source": {
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
    }


def ingest_fixture(
    connection: sqlite3.Connection,
    tmp_path: Path,
    records: dict[str, dict[str, object]] | None = None,
) -> None:
    selected_records = records if records is not None else fixture_records()
    archive_path = tmp_path / "substitution-fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for filename, source in DATASET_FILES.items():
            archive.writestr(
                filename,
                json.dumps(selected_records if source == "ar" else {}),
            )
    ingest_dataset_archive(connection, archive_path)


def test_direct_baked_substitution_uses_real_supporting_records_and_preserves_source(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    ingest_fixture(connection, tmp_path)

    proposal = propose_substitutions(
        connection,
        "ar",
        "apple-muffin-source",
        "eggs",
        replacement_ingredients=("ground flaxseed",),
    )

    assert proposal is not None
    assert proposal.original_recipe.recipe_id == "apple-muffin-source"
    assert proposal.original_title == "Apple Breakfast Muffins"
    assert proposal.ingredient_to_replace == "2 large eggs"
    assert proposal.original_ingredient_list[0] == "2 large eggs"
    option = proposal.options[0]
    assert option.evidence_strength == "direct"
    assert option.sufficiently_supported
    assert option.revised_ingredient_list == (
        "ground flaxseed",
        "2 cups flour",
        "1 cup apple",
        "1/2 cup sugar",
        "1 teaspoon vanilla",
    )
    assert {
        reference.recipe_id for reference in option.cooccurrence_recipe_ids
    } == {"egg-flax-cooccurrence"}
    assert {record.reference.recipe_id for record in option.supporting_recipes} == {
        "apple-muffin-flax",
        "apple-muffin-flax-second",
    }
    assert all(
        connection.execute(
            "SELECT 1 FROM recipes WHERE source = ? AND recipe_id = ?",
            (record.reference.source, record.reference.recipe_id),
        ).fetchone()
        is not None
        for record in option.supporting_recipes
    )
    assert all(
        "Mix" in record.instruction_excerpt[0]
        for record in option.supporting_recipes
    )
    assert option.instruction_adjustments[0].original_instruction.startswith(
        "1. Mix eggs"
    )
    assert option.instruction_adjustments[0].suggested_instruction is None
    assert not option.ingredient_roles[0].certainty.startswith("certain")
    assert "quantities" in " ".join(option.remaining_uncertainty)


def test_indirect_and_weak_evidence_are_not_reliable_adaptations(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    ingest_fixture(connection, tmp_path)

    indirect = propose_substitutions(
        connection,
        "ar",
        "apple-muffin-source",
        "egg",
        replacement_ingredients=("honey",),
    )
    weak = propose_substitutions(
        connection,
        "ar",
        "apple-muffin-source",
        "egg",
        replacement_ingredients=("turmeric",),
    )

    assert indirect is not None and indirect.options[0].evidence_strength == "indirect"
    assert indirect.options[0].revised_ingredient_list is None
    assert weak is not None and weak.options[0].evidence_strength == "weak"
    assert weak.options[0].supporting_recipes[0].reference.recipe_id == "turmeric-soup"
    assert weak.options[0].sufficiently_supported is False


def test_no_evidence_and_unparseable_replacements_are_explicit(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    ingest_fixture(connection, tmp_path)

    proposal = propose_substitutions(
        connection,
        "ar",
        "apple-muffin-source",
        "egg",
        replacement_ingredients=("unicorn foam",),
    )

    assert proposal is not None
    assert proposal.options[0].evidence_strength == "no_evidence"
    assert not proposal.options[0].supporting_recipes
    assert proposal.options[0].revised_ingredient_list is None
    with pytest.raises(ValueError, match="could not be parsed"):
        propose_substitutions(
            connection,
            "ar",
            "apple-muffin-source",
            "egg",
            replacement_ingredients=("!!!",),
        )


def test_multiple_direct_alternatives_are_reported_as_conflicting(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    ingest_fixture(connection, tmp_path)

    proposal = propose_substitutions(
        connection,
        "ar",
        "apple-muffin-source",
        "egg",
        replacement_ingredients=("ground flaxseed", "banana"),
    )

    assert proposal is not None
    assert proposal.status == "unsupported"
    assert len(proposal.options) == 2
    assert all(option.evidence_strength == "direct" for option in proposal.options)
    assert all(option.conflicting_evidence for option in proposal.options)
    assert all(not option.sufficiently_supported for option in proposal.options)
    assert all(option.revised_ingredient_list is None for option in proposal.options)


def test_automatic_alternative_discovery_and_preferences_are_deterministic(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    ingest_fixture(connection, tmp_path)

    proposal = propose_substitutions(
        connection,
        "ar",
        "apple-muffin-source",
        "egg",
        preferences=("banana",),
    )

    assert proposal is not None
    names = {option.replacement_canonical_name for option in proposal.options}
    assert {"ground flaxseed", "banana", "honey"} <= names
    assert proposal.options[0].replacement_canonical_name == "banana"
    assert all(option.conflicting_evidence for option in proposal.options[:2])


def test_savory_scenario_enforces_exclusions_and_dairy_free_restrictions(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    ingest_fixture(connection, tmp_path)

    tofu = propose_substitutions(
        connection,
        "ar",
        "savory-soup-source",
        "chicken breast",
        replacement_ingredients=("tofu",),
        excluded_ingredients=("tofu",),
    )
    oat_milk = propose_substitutions(
        connection,
        "ar",
        "milk-cake-source",
        "milk",
        replacement_ingredients=("oat milk",),
        dietary_restrictions=("dairy-free",),
    )

    assert tofu is not None and tofu.status == "unsupported"
    assert tofu.options[0].evidence_strength == "direct"
    assert tofu.options[0].revised_ingredient_list is None
    assert tofu.options[0].rejection_reason is not None
    assert oat_milk is not None and oat_milk.status == "supported"
    assert oat_milk.options[0].revised_ingredient_list is not None
    assert not any(
        parse_ingredient(line).canonical_name
        in {"milk", "butter", "cream", "cheese"}
        for line in oat_milk.options[0].revised_ingredient_list
    )
    assert "allergy-safety" in " ".join(oat_milk.limitations)


def test_milk_free_restriction_does_not_allow_plant_milk(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    ingest_fixture(connection, tmp_path)

    proposal = propose_substitutions(
        connection,
        "ar",
        "milk-cake-source",
        "milk",
        replacement_ingredients=("oat milk",),
        dietary_restrictions=("milk-free",),
    )

    assert proposal is not None
    assert proposal.status == "unsupported"
    assert proposal.options[0].sufficiently_supported is False
    assert proposal.options[0].revised_ingredient_list is None
    assert "restricted ingredient" in (proposal.options[0].rejection_reason or "")


def test_exclusion_is_checked_against_every_retained_ingredient(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    records = fixture_records()
    ingredients = records["apple-muffin-source"]["ingredients"]
    assert isinstance(ingredients, list)
    ingredients.append("1 tablespoon peanut butter")
    ingest_fixture(connection, tmp_path, records)

    proposal = propose_substitutions(
        connection,
        "ar",
        "apple-muffin-source",
        "egg",
        replacement_ingredients=("ground flaxseed",),
        excluded_ingredients=("peanut",),
    )

    assert proposal is not None
    option = proposal.options[0]
    assert option.evidence_strength == "direct"
    assert not option.sufficiently_supported
    assert option.revised_ingredient_list is None
    assert "restricted ingredient" in (option.rejection_reason or "")


def test_ambiguous_source_ingredient_prevents_restricted_adaptation(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    records = fixture_records()
    ingredients = records["apple-muffin-source"]["ingredients"]
    assert isinstance(ingredients, list)
    ingredients.append("!!!")
    ingest_fixture(connection, tmp_path, records)

    proposal = propose_substitutions(
        connection,
        "ar",
        "apple-muffin-source",
        "egg",
        replacement_ingredients=("ground flaxseed",),
        excluded_ingredients=("peanut",),
    )

    assert proposal is not None
    assert proposal.options[0].revised_ingredient_list is None
    assert proposal.options[0].rejection_reason is not None
    assert "ambiguous" in proposal.options[0].rejection_reason


def test_missing_recipe_and_missing_or_ambiguous_target_are_handled(
    connection: sqlite3.Connection, tmp_path: Path
) -> None:
    ingest_fixture(connection, tmp_path)

    assert (
        propose_substitutions(
            connection, "ar", "not-a-recipe", "egg"
        )
        is None
    )
    with pytest.raises(ValueError, match="not present"):
        propose_substitutions(
            connection, "ar", "apple-muffin-source", "chocolate"
        )

    repeated = fixture_records()
    repeated["apple-muffin-source"]["ingredients"] = [
        "2 eggs",
        "1 egg",
        "2 cups flour",
    ]
    ingest_fixture(connection, tmp_path, repeated)
    with pytest.raises(ValueError, match="more than once"):
        propose_substitutions(
            connection, "ar", "apple-muffin-source", "egg"
        )
