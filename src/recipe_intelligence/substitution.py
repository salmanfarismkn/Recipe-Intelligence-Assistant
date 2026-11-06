"""Evidence retrieval and conservative local ingredient-substitution proposals."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, replace
from typing import Literal, Sequence

from recipe_intelligence.matching import INGREDIENT_ALIASES, parse_ingredient
from recipe_intelligence.normalization import normalize_ingredient
from recipe_intelligence.retrieval import get_recipe

EvidenceStrength = Literal["direct", "indirect", "weak", "no_evidence"]

_PREPARATION_METHODS = {
    "bake": {"bake", "bakes", "baked", "baking"},
    "boil": {"boil", "boils", "boiled", "boiling"},
    "braise": {"braise", "braises", "braised", "braising"},
    "chill": {"chill", "chills", "chilled", "chilling"},
    "fry": {"fry", "fries", "fried", "frying"},
    "grill": {"grill", "grills", "grilled", "grilling"},
    "mix": {"mix", "mixes", "mixed", "mixing"},
    "roast": {"roast", "roasts", "roasted", "roasting"},
    "saute": {"saute", "sautes", "sauteed", "sauteing"},
    "simmer": {"simmer", "simmers", "simmered", "simmering"},
    "steam": {"steam", "steams", "steamed", "steaming"},
    "stir": {"stir", "stirs", "stirred", "stirring"},
}
_TITLE_STOP_WORDS = {
    "a",
    "and",
    "best",
    "easy",
    "for",
    "of",
    "recipe",
    "the",
    "with",
}
_DAIRY_FREE_INGREDIENTS = frozenset(
    {
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
    }
)
_PLANT_MILKS = frozenset(
    {
        "almond milk",
        "cashew milk",
        "coconut milk",
        "oat milk",
        "rice milk",
        "soy milk",
    }
)


@dataclass(frozen=True)
class RecipeReference:
    source: str
    recipe_id: str


@dataclass(frozen=True)
class IngredientRoleAssessment:
    ingredient: str
    role_candidates: tuple[str, ...]
    method: str
    certainty: str


@dataclass(frozen=True)
class SupportingRecipe:
    reference: RecipeReference
    title: str | None
    replacement_ingredient_lines: tuple[str, ...]
    shared_ingredients: tuple[str, ...]
    shared_title_terms: tuple[str, ...]
    shared_preparation_methods: tuple[str, ...]
    ingredient_overlap: float
    instruction_excerpt: tuple[str, ...]


@dataclass(frozen=True)
class InstructionAdjustment:
    original_instruction: str
    adjustment_needed: str
    suggested_instruction: None = None


@dataclass(frozen=True)
class SubstitutionOption:
    replacement_ingredient: str
    replacement_canonical_name: str
    evidence_strength: EvidenceStrength
    sufficiently_supported: bool
    supporting_recipes: tuple[SupportingRecipe, ...]
    cooccurrence_recipe_ids: tuple[RecipeReference, ...]
    conflicting_evidence: bool
    reasoning: str
    rejection_reason: str | None
    ingredient_roles: tuple[IngredientRoleAssessment, ...]
    revised_ingredient_list: tuple[str, ...] | None
    instruction_adjustments: tuple[InstructionAdjustment, ...]
    expected_trade_offs: tuple[str, ...]
    remaining_uncertainty: tuple[str, ...]


@dataclass(frozen=True)
class SubstitutionProposal:
    original_recipe: RecipeReference
    original_title: str | None
    ingredient_to_replace: str
    original_ingredient_list: tuple[str, ...]
    options: tuple[SubstitutionOption, ...]
    status: Literal["supported", "unsupported"]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _CandidateRecipe:
    reference: RecipeReference
    title: str | None
    instructions: str | None
    instruction_step_count: int | None
    ingredient_names: tuple[str, ...]
    replacement_lines: tuple[str, ...]
    fully_parsed: bool


def _role_candidates(canonical_name: str) -> IngredientRoleAssessment:
    tokens = set(canonical_name.split())
    roles: list[str] = []
    if tokens & {"egg", "flaxseed", "gelatin", "chia"}:
        roles.append("binding")
    if tokens & {"water", "milk", "broth", "stock", "juice", "yogurt"}:
        roles.append("moisture")
    if tokens & {"butter", "oil", "shortening", "lard", "margarine", "cream"}:
        roles.append("fat")
    if tokens & {"flour", "cornstarch", "starch", "gelatin"}:
        roles.append("thickening")
    if tokens & {
        "garlic",
        "onion",
        "salt",
        "pepper",
        "herb",
        "spice",
        "vanilla",
        "lemon",
        "lime",
    }:
        roles.append("flavor")
    if tokens & {
        "nut",
        "almond",
        "walnut",
        "seed",
        "carrot",
        "celery",
        "apple",
        "chocolate",
        "oat",
    }:
        roles.append("texture")
    if tokens & {"egg", "flour", "bread", "oat", "gelatin"}:
        roles.append("structure")
    return IngredientRoleAssessment(
        ingredient=canonical_name,
        role_candidates=tuple(roles),
        method="heuristic ingredient-name lexicon",
        certainty="heuristic; not established by the recipe record",
    )


def _preparation_methods(instructions: str | None) -> frozenset[str]:
    words = set(normalize_ingredient(instructions or "").split())
    return frozenset(
        method
        for method, forms in _PREPARATION_METHODS.items()
        if words.intersection(forms)
    )


def _title_terms(title: str | None) -> frozenset[str]:
    return frozenset(
        token
        for token in normalize_ingredient(title or "").split()
        if token not in _TITLE_STOP_WORDS and len(token) > 2
    )


def _instruction_excerpt(
    instructions: str | None,
    ingredient_names: Sequence[str],
    method_names: Sequence[str],
) -> tuple[str, ...]:
    if not instructions:
        return ()
    search_terms = set(method_names)
    for name in ingredient_names:
        search_terms.update(name.split())
    excerpts = []
    for line in instructions.splitlines():
        normalized = normalize_ingredient(line)
        if normalized and any(
            re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized)
            for term in search_terms
        ):
            excerpts.append(line)
        if len(excerpts) == 4:
            break
    return tuple(excerpts)


def _contains_any(
    canonical_name: str | None,
    normalized_text: str | None,
    blocked: frozenset[str],
    *,
    allow_plant_milks: bool = False,
) -> bool:
    text = normalize_ingredient(
        f"{canonical_name or ''} {normalized_text or ''}"
    )
    if allow_plant_milks and "milk" in blocked:
        for plant_milk in _PLANT_MILKS:
            text = re.sub(rf"\b{re.escape(plant_milk)}\b", " ", text)
    padded_text = f" {text} "
    return any(f" {name} " in padded_text for name in blocked)


def _normalize_restrictions(
    excluded_ingredients: Sequence[str],
    dietary_restrictions: Sequence[str],
) -> tuple[frozenset[str], bool]:
    blocked: set[str] = set()
    explicit_milk_exclusion = False
    strict_milk_restriction = False
    allow_plant_milks = False
    for value in excluded_ingredients:
        parsed = parse_ingredient(value)
        if parsed.canonical_name is None:
            raise ValueError(f"excluded ingredient could not be parsed: {value!r}")
        blocked.add(parsed.canonical_name)
        explicit_milk_exclusion |= parsed.canonical_name == "milk"

    for restriction in dietary_restrictions:
        if not isinstance(restriction, str):
            raise TypeError("dietary restrictions must be strings")
        normalized = normalize_ingredient(restriction)
        if normalized in {"dairy free", "dairy-free"}:
            blocked.update(_DAIRY_FREE_INGREDIENTS)
            allow_plant_milks = True
        elif normalized.endswith(" free"):
            ingredient_text = normalized.removesuffix(" free")
            parsed = parse_ingredient(ingredient_text)
            if parsed.canonical_name is None:
                raise ValueError(
                    f"dietary restriction could not be interpreted: {restriction!r}"
                )
            blocked.add(parsed.canonical_name)
            strict_milk_restriction |= parsed.canonical_name == "milk"
        else:
            raise ValueError(
                "dietary restrictions must use a supported '<ingredient>-free' "
                "form, such as 'dairy-free' or 'peanut-free'"
            )
    return (
        frozenset(blocked),
        allow_plant_milks and not explicit_milk_exclusion and not strict_milk_restriction,
    )


def _recipe_candidates(
    connection: sqlite3.Connection,
    replacement_name: str,
    replaced_name: str,
    base_ingredients: tuple[str, ...],
    limit: int,
) -> tuple[_CandidateRecipe, ...]:
    if not base_ingredients:
        return ()
    placeholders = ",".join("?" for _ in base_ingredients)
    rows = connection.execute(
        f"""
        WITH related AS (
            SELECT
                r.source,
                r.recipe_id,
                r.title,
                r.instructions,
                r.instruction_step_count,
                COUNT(DISTINCT shared.canonical_name) AS shared_count
            FROM recipe_ingredients AS alternative
            JOIN recipes AS r
              ON r.source = alternative.source
             AND r.recipe_id = alternative.recipe_id
            JOIN recipe_ingredients AS shared
              ON shared.source = r.source
             AND shared.recipe_id = r.recipe_id
             AND shared.canonical_name IN ({placeholders})
            WHERE alternative.canonical_name = ?
              AND r.ingredients_valid = 1
              AND NOT EXISTS (
                  SELECT 1
                  FROM recipe_ingredients AS replaced
                  WHERE replaced.source = r.source
                    AND replaced.recipe_id = r.recipe_id
                    AND replaced.canonical_name = ?
              )
            GROUP BY r.source, r.recipe_id
            ORDER BY shared_count DESC, r.source, r.recipe_id
            LIMIT ?
        )
        SELECT
            related.source,
            related.recipe_id,
            related.title,
            related.instructions,
            related.instruction_step_count,
            related.shared_count,
            ingredients.position,
            ingredients.original_text,
            ingredients.canonical_name
        FROM related
        JOIN recipe_ingredients AS ingredients
          ON ingredients.source = related.source
         AND ingredients.recipe_id = related.recipe_id
        ORDER BY related.shared_count DESC, related.source,
                 related.recipe_id, ingredients.position
        """,
        (*base_ingredients, replacement_name, replaced_name, limit),
    ).fetchall()
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (row["source"], row["recipe_id"])
        if key not in grouped:
            grouped[key] = {
                "title": row["title"],
                "instructions": row["instructions"],
                "step_count": row["instruction_step_count"],
                "ingredient_names": set(),
                "replacement_lines": [],
                "fully_parsed": True,
            }
        if row["canonical_name"] is None or not isinstance(row["original_text"], str):
            grouped[key]["fully_parsed"] = False
        if row["canonical_name"] is not None:
            grouped[key]["ingredient_names"].add(row["canonical_name"])
        if (
            row["canonical_name"] == replacement_name
            and isinstance(row["original_text"], str)
        ):
            grouped[key]["replacement_lines"].append(row["original_text"])

    return tuple(
        _CandidateRecipe(
            reference=RecipeReference(source=source, recipe_id=recipe_id),
            title=record["title"] if isinstance(record["title"], str) else None,
            instructions=(
                record["instructions"]
                if isinstance(record["instructions"], str)
                else None
            ),
            instruction_step_count=(
                record["step_count"]
                if isinstance(record["step_count"], int)
                else None
            ),
            ingredient_names=tuple(sorted(record["ingredient_names"])),
            replacement_lines=tuple(record["replacement_lines"]),
            fully_parsed=bool(record["fully_parsed"]),
        )
        for (source, recipe_id), record in grouped.items()
    )


def _fallback_examples(
    connection: sqlite3.Connection, replacement_name: str, replaced_name: str
) -> tuple[SupportingRecipe, ...]:
    rows = connection.execute(
        """
        SELECT DISTINCT
            recipes.source, recipes.recipe_id, recipes.title, recipes.instructions
        FROM recipe_ingredients AS candidate
        JOIN recipes
          ON recipes.source = candidate.source
         AND recipes.recipe_id = candidate.recipe_id
        WHERE candidate.canonical_name = ?
          AND recipes.ingredients_valid = 1
          AND NOT EXISTS (
              SELECT 1 FROM recipe_ingredients
              WHERE recipe_ingredients.source = recipes.source
                AND recipe_ingredients.recipe_id = recipes.recipe_id
                AND recipe_ingredients.canonical_name = ?
          )
        ORDER BY recipes.source, recipes.recipe_id
        LIMIT 3
        """,
        (replacement_name, replaced_name),
    ).fetchall()
    examples = []
    for row in rows:
        ingredient_rows = connection.execute(
            """
            SELECT original_text, canonical_name
            FROM recipe_ingredients
            WHERE source = ? AND recipe_id = ?
            ORDER BY position
            """,
            (row["source"], row["recipe_id"]),
        ).fetchall()
        ingredient_names = tuple(
            item["canonical_name"]
            for item in ingredient_rows
            if item["canonical_name"] is not None
        )
        replacement_lines = tuple(
            item["original_text"]
            for item in ingredient_rows
            if item["canonical_name"] == replacement_name
            and isinstance(item["original_text"], str)
        )
        examples.append(
            SupportingRecipe(
                reference=RecipeReference(row["source"], row["recipe_id"]),
                title=row["title"],
                replacement_ingredient_lines=replacement_lines,
                shared_ingredients=(),
                shared_title_terms=(),
                shared_preparation_methods=(),
                ingredient_overlap=0.0,
                instruction_excerpt=_instruction_excerpt(
                    row["instructions"], ingredient_names, ()
                ),
            )
        )
    return tuple(examples)


def _cooccurrence_ids(
    connection: sqlite3.Connection, replaced_name: str, replacement_name: str
) -> tuple[RecipeReference, ...]:
    rows = connection.execute(
        """
        SELECT DISTINCT source, recipe_id
        FROM recipe_ingredients
        WHERE canonical_name IN (?, ?)
        GROUP BY source, recipe_id
        HAVING COUNT(DISTINCT canonical_name) = 2
        ORDER BY source, recipe_id
        LIMIT 20
        """,
        (replaced_name, replacement_name),
    ).fetchall()
    return tuple(RecipeReference(row["source"], row["recipe_id"]) for row in rows)


def _classify_examples(
    candidates: tuple[_CandidateRecipe, ...],
    *,
    source_title: str | None,
    source_instructions: str | None,
    source_steps: int | None,
    replaced_name: str,
    replacement_name: str,
    base_ingredients: frozenset[str],
    source_fully_parsed: bool,
) -> tuple[EvidenceStrength, tuple[SupportingRecipe, ...]]:
    source_methods = _preparation_methods(source_instructions)
    source_title_terms = _title_terms(source_title)
    classified: list[tuple[EvidenceStrength, SupportingRecipe]] = []
    for candidate in candidates:
        candidate_ingredients = frozenset(candidate.ingredient_names)
        shared = tuple(sorted(base_ingredients & candidate_ingredients))
        overlap = len(shared) / len(base_ingredients) if base_ingredients else 0.0
        shared_methods = tuple(
            sorted(source_methods & _preparation_methods(candidate.instructions))
        )
        shared_titles = tuple(
            sorted(source_title_terms & _title_terms(candidate.title))
        )
        if (
            source_fully_parsed
            and candidate.fully_parsed
            and len(shared) >= 2
            and overlap >= 0.75
            and shared_methods
            and (
                source_steps is not None
                and candidate.instruction_step_count == source_steps
                or shared_titles
            )
        ):
            strength: EvidenceStrength = "direct"
        elif (
            source_fully_parsed
            and candidate.fully_parsed
            and len(shared) >= 2
            and overlap >= 0.5
            and (shared_methods or shared_titles)
        ):
            strength = "indirect"
        else:
            strength = "weak"
        classified.append(
            (
                strength,
                SupportingRecipe(
                    reference=candidate.reference,
                    title=candidate.title,
                    replacement_ingredient_lines=candidate.replacement_lines,
                    shared_ingredients=shared,
                    shared_title_terms=shared_titles,
                    shared_preparation_methods=shared_methods,
                    ingredient_overlap=overlap,
                    instruction_excerpt=_instruction_excerpt(
                        candidate.instructions,
                        (replaced_name, replacement_name),
                        shared_methods,
                    ),
                ),
            )
        )
    rank = {"direct": 3, "indirect": 2, "weak": 1, "no_evidence": 0}
    classified.sort(
        key=lambda item: (
            -rank[item[0]],
            -item[1].ingredient_overlap,
            item[1].reference.source,
            item[1].reference.recipe_id,
        )
    )
    if not classified:
        return "no_evidence", ()
    strongest = classified[0][0]
    strongest_examples = tuple(
        evidence for strength, evidence in classified if strength == strongest
    )
    return strongest, strongest_examples


def _instruction_adjustments(
    instructions: str | None, replaced_name: str
) -> tuple[InstructionAdjustment, ...]:
    if not instructions:
        return ()
    variants = {replaced_name}
    variants.update(
        alias for alias, canonical in INGREDIENT_ALIASES.items()
        if canonical == replaced_name
    )
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(term) for term in sorted(
            variants, key=lambda value: (-len(value), value)
        )) + r")(?!\w)",
        re.IGNORECASE,
    )
    return tuple(
        InstructionAdjustment(
            original_instruction=line,
            adjustment_needed=(
                "This source instruction names the ingredient being replaced. "
                "Adapt the step to the replacement; local records do not establish "
                "a safe exact rewrite or quantity."
            ),
        )
        for line in instructions.splitlines()
        if pattern.search(line)
    )


def _validate_sequence(values: Sequence[str], label: str) -> None:
    if isinstance(values, str):
        raise TypeError(f"{label} must be a sequence of strings, not a string")
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{label} must contain only strings")


def propose_substitutions(
    connection: sqlite3.Connection,
    source: str,
    recipe_id: str,
    ingredient_to_replace: str,
    *,
    replacement_ingredients: Sequence[str] = (),
    excluded_ingredients: Sequence[str] = (),
    dietary_restrictions: Sequence[str] = (),
    preferences: Sequence[str] = (),
    limit: int = 10,
) -> SubstitutionProposal | None:
    """Return local-evidence options; only strong, non-conflicting options adapt."""
    for values, label in (
        (replacement_ingredients, "replacement ingredients"),
        (excluded_ingredients, "excluded ingredients"),
        (dietary_restrictions, "dietary restrictions"),
        (preferences, "preferences"),
    ):
        _validate_sequence(values, label)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("limit must be an integer from 1 to 20")
    if len(replacement_ingredients) > limit:
        raise ValueError("replacement ingredient count cannot exceed limit")
    if not isinstance(ingredient_to_replace, str):
        raise TypeError("ingredient to replace must be a string")
    blocked, allow_plant_milks = _normalize_restrictions(
        excluded_ingredients, dietary_restrictions
    )
    preferred_names: set[str] = set()
    for preference in preferences:
        parsed = parse_ingredient(preference)
        if parsed.canonical_name is None:
            raise ValueError(f"preferred ingredient could not be parsed: {preference!r}")
        preferred_names.add(parsed.canonical_name)

    detail = get_recipe(connection, source, recipe_id)
    if detail is None:
        return None
    raw_ingredients = detail.original_record.get("ingredients")
    if not isinstance(raw_ingredients, list) or not all(
        isinstance(value, str) for value in raw_ingredients
    ):
        return SubstitutionProposal(
            original_recipe=RecipeReference(source, recipe_id),
            original_title=detail.original_record.get("title"),
            ingredient_to_replace=ingredient_to_replace,
            original_ingredient_list=tuple(
                value for value in raw_ingredients if isinstance(value, str)
            )
            if isinstance(raw_ingredients, list)
            else (),
            options=(),
            status="unsupported",
            limitations=(
                "The source recipe has a missing or malformed ingredient list; "
                "no adaptation was attempted.",
            ),
        )

    parsed_target = parse_ingredient(ingredient_to_replace)
    if parsed_target.canonical_name is None:
        raise ValueError(
            f"ingredient to replace could not be parsed: {ingredient_to_replace!r}"
        )
    target_rows = connection.execute(
        """
        SELECT position, original_text, canonical_name, normalized_text
        FROM recipe_ingredients
        WHERE source = ? AND recipe_id = ?
          AND canonical_name = ?
        ORDER BY position
        """,
        (source, recipe_id, parsed_target.canonical_name),
    ).fetchall()
    if not target_rows:
        raise ValueError("ingredient to replace is not present in the requested recipe")
    if len(target_rows) != 1:
        raise ValueError(
            "ingredient to replace occurs more than once; identify one occurrence "
            "before requesting a substitution"
        )
    target_row = target_rows[0]
    replaced_name = parsed_target.canonical_name
    ingredient_rows = connection.execute(
        """
        SELECT canonical_name
        FROM recipe_ingredients
        WHERE source = ? AND recipe_id = ?
        """,
        (source, recipe_id),
    ).fetchall()
    source_fully_parsed = all(row["canonical_name"] is not None for row in ingredient_rows)
    original_names = {
        row["canonical_name"]
        for row in ingredient_rows
        if row["canonical_name"] is not None
    }
    base_ingredients = tuple(sorted(original_names - {replaced_name}))

    explicit_replacements: list[tuple[str, str]] = []
    for replacement in replacement_ingredients:
        parsed = parse_ingredient(replacement)
        if parsed.canonical_name is None:
            raise ValueError(f"replacement ingredient could not be parsed: {replacement!r}")
        if parsed.canonical_name == replaced_name:
            raise ValueError("replacement ingredient is the same as the ingredient replaced")
        if parsed.canonical_name in original_names:
            raise ValueError("replacement ingredient already occurs in the original recipe")
        if parsed.canonical_name not in {
            name for _, name in explicit_replacements
        }:
            explicit_replacements.append((replacement, parsed.canonical_name))

    source_instructions = detail.original_record.get("instructions")
    source_instructions = (
        source_instructions if isinstance(source_instructions, str) else None
    )
    source_title = detail.original_record.get("title")
    source_title = source_title if isinstance(source_title, str) else None
    source_steps_row = connection.execute(
        """
        SELECT instruction_step_count FROM recipes
        WHERE source = ? AND recipe_id = ?
        """,
        (source, recipe_id),
    ).fetchone()
    source_steps = (
        source_steps_row["instruction_step_count"]
        if source_steps_row is not None
        and isinstance(source_steps_row["instruction_step_count"], int)
        else None
    )

    if explicit_replacements:
        candidate_names = explicit_replacements
    else:
        existing_placeholders = ",".join("?" for _ in original_names)
        candidate_rows = connection.execute(
            f"""
            SELECT candidate.canonical_name,
                   COUNT(DISTINCT candidate.source || ':' || candidate.recipe_id)
                       AS related_recipe_count
            FROM recipe_ingredients AS replaced
            JOIN recipe_ingredients AS candidate
              ON candidate.source = replaced.source
             AND candidate.recipe_id = replaced.recipe_id
            JOIN recipes
              ON recipes.source = candidate.source
             AND recipes.recipe_id = candidate.recipe_id
            WHERE replaced.canonical_name = ?
              AND candidate.canonical_name IS NOT NULL
              AND candidate.canonical_name NOT IN ({existing_placeholders})
              AND recipes.ingredients_valid = 1
            GROUP BY candidate.canonical_name
            ORDER BY related_recipe_count DESC, candidate.canonical_name
            LIMIT ?
            """,
            (replaced_name, *sorted(original_names), limit),
        ).fetchall()
        candidate_names = [
            (row["canonical_name"], row["canonical_name"])
            for row in candidate_rows
        ]

    draft_options: list[SubstitutionOption] = []
    for replacement_display, replacement_name in candidate_names:
        candidates = _recipe_candidates(
            connection,
            replacement_name,
            replaced_name,
            tuple(base_ingredients),
            20,
        )
        evidence_strength, supporting = _classify_examples(
            candidates,
            source_title=source_title,
            source_instructions=source_instructions,
            source_steps=source_steps,
            replaced_name=replaced_name,
            replacement_name=replacement_name,
            base_ingredients=frozenset(base_ingredients),
            source_fully_parsed=source_fully_parsed,
        )
        if evidence_strength == "no_evidence":
            supporting = _fallback_examples(
                connection, replacement_name, replaced_name
            )
            if supporting:
                evidence_strength = "weak"

        cooccurrences = _cooccurrence_ids(
            connection, replaced_name, replacement_name
        )
        original_lines = list(raw_ingredients)
        original_lines[target_row["position"]] = replacement_name
        restriction_failure = None
        if blocked:
            if replacement_name in blocked or any(
                _contains_any(
                    parse_ingredient(line).canonical_name,
                    normalize_ingredient(line),
                    blocked,
                    allow_plant_milks=allow_plant_milks,
                )
                for line in original_lines
            ):
                restriction_failure = (
                    "The complete revised ingredient list contains a restricted "
                    "ingredient or an ingredient whose parsed identity is uncertain."
                )
            elif any(
                parse_ingredient(line).canonical_name is None
                for line in original_lines
            ):
                restriction_failure = (
                    "At least one revised ingredient is ambiguous, so the hard "
                    "restriction cannot be verified."
                )

        sufficiently_supported = evidence_strength == "direct"
        if restriction_failure:
            sufficiently_supported = False
        revised = (
            tuple(original_lines)
            if sufficiently_supported and not restriction_failure
            else None
        )
        reasoning = {
            "direct": (
                "At least one local recipe excludes the replaced ingredient, uses "
                "the proposed alternative, shares at least 75% of the original "
                "recipe's other parsed ingredients (minimum two), and shares a "
                "recognized preparation method plus matching step count or title "
                "context. This is comparable-recipe evidence, not proof of an "
                "interchangeable quantity or guaranteed outcome."
            ),
            "indirect": (
                "A related recipe uses the alternative and shares at least half "
                "of the original recipe's other parsed ingredients (minimum two), "
                "with a shared preparation method or title term; the match is not "
                "close enough for a reliable adaptation."
            ),
            "weak": (
                "The alternative occurs in the dataset, but no sufficiently "
                "comparable recipe context was retrieved."
            ),
            "no_evidence": (
                "No adequate local comparison context for this alternative was found."
            ),
        }[evidence_strength]
        if cooccurrences:
            reasoning += (
                " Separate recipes contain both ingredients; co-occurrence is "
                "reported but is not treated as replacement evidence."
            )
        if restriction_failure:
            reasoning += f" Adaptation refused: {restriction_failure}"
        draft_options.append(
            SubstitutionOption(
                replacement_ingredient=replacement_display,
                replacement_canonical_name=replacement_name,
                evidence_strength=evidence_strength,
                sufficiently_supported=sufficiently_supported,
                supporting_recipes=supporting,
                cooccurrence_recipe_ids=cooccurrences,
                conflicting_evidence=False,
                reasoning=reasoning,
                rejection_reason=restriction_failure,
                ingredient_roles=(
                    _role_candidates(replaced_name),
                    _role_candidates(replacement_name),
                ),
                revised_ingredient_list=revised,
                instruction_adjustments=_instruction_adjustments(
                    source_instructions, replaced_name
                ),
                expected_trade_offs=(
                    "Taste, texture, and preparation trade-offs are not established "
                    "by the retrieved recipe records.",
                ),
                remaining_uncertainty=(
                    "Ingredient quantities are not compared or transferred.",
                    "Similar recipe composition does not prove that the alternative "
                    "is interchangeable.",
                    "No medical or allergy-safety conclusion is made.",
                ),
            )
        )

    direct_names = {
        option.replacement_canonical_name
        for option in draft_options
        if option.evidence_strength == "direct"
    }
    conflict = len(direct_names) > 1
    options = []
    for option in draft_options:
        if not conflict:
            options.append(option)
            continue
        options.append(
            replace(
                option,
                sufficiently_supported=False,
                conflicting_evidence=option.evidence_strength == "direct",
                reasoning=(
                    option.reasoning
                    + " Multiple alternatives meet the direct-evidence rule; "
                    "the local evidence does not distinguish a single choice."
                ),
                revised_ingredient_list=None,
            )
        )
    preferred = preferred_names
    options.sort(
        key=lambda option: (
            -int(option.sufficiently_supported),
            -int(option.replacement_canonical_name in preferred),
            {
                "direct": 0,
                "indirect": 1,
                "weak": 2,
                "no_evidence": 3,
            }[option.evidence_strength],
            option.replacement_canonical_name,
        )
    )
    return SubstitutionProposal(
        original_recipe=RecipeReference(source, recipe_id),
        original_title=source_title,
        ingredient_to_replace=target_row["original_text"],
        original_ingredient_list=tuple(raw_ingredients),
        options=tuple(options),
        status=(
            "supported"
            if any(option.sufficiently_supported for option in options)
            else "unsupported"
        ),
        limitations=(
            "Local data contains no explicit substitution labels; evidence is "
            "classified from comparable ingredient lists, instruction methods, "
            "and title terms.",
            "A revised ingredient list never transfers a quantity. The replacement "
            "line is the ingredient name only.",
            "Source instructions are preserved; relevant lines are flagged for "
            "manual adaptation rather than rewritten without evidence.",
            "Ingredient-text restrictions are not a medical or allergy-safety "
            "guarantee.",
        ),
    )
