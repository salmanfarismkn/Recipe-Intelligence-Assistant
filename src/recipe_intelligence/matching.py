"""Conservative ingredient parsing and deterministic recipe matching."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Sequence

from recipe_intelligence.normalization import normalize_ingredient

INGREDIENT_ALIASES = {
    "chickpeas": "chickpea",
    "chicken breasts": "chicken breast",
    "courgette": "zucchini",
    "courgettes": "zucchini",
    "eggs": "egg",
    "garbanzo beans": "chickpea",
    "green onions": "green onion",
    "chicken breast halves": "chicken breast",
    "lemons": "lemon",
    "limes": "lime",
    "onions": "onion",
    "peanuts": "peanut",
    "potatoes": "potato",
    "scallions": "green onion",
    "spring onions": "green onion",
    "tomatoes": "tomato",
    "zucchinis": "zucchini",
}
_RANKING_PREFERENCE_ALIASES = {
    "fewer additional": "fewer_additional",
    "fewer additional ingredients": "fewer_additional",
    "fewer missing": "fewer_missing",
    "fewer missing ingredients": "fewer_missing",
    "fewer steps": "fewer_steps",
    "fewer instruction steps": "fewer_steps",
    "fewer_missing": "fewer_missing",
    "fewer_additional": "fewer_additional",
    "fewer_steps": "fewer_steps",
}

_NUMERIC_AMOUNT = re.compile(
    r"^\s*(?:(?P<mixed>\d+\s+\d+\s*/\s*\d+)"
    r"|(?P<fraction>\d+\s*/\s*\d+)"
    r"|(?P<decimal>\d+(?:\.\d+)?))(?=\s|$)"
)
_RANGE_AMOUNT = re.compile(
    r"^\s*\d+(?:\s*/\s*\d+)?\s*(?:-|to)\s*"
    r"\d+(?:\s*/\s*\d+)?(?=\s|$)",
    re.IGNORECASE,
)
_WORD_AMOUNTS = {
    "a": Fraction(1),
    "an": Fraction(1),
    "one": Fraction(1),
    "two": Fraction(2),
    "three": Fraction(3),
    "four": Fraction(4),
    "five": Fraction(5),
}
_WORD_AMOUNT = re.compile(r"^\s*(?P<amount>a|an|one|two|three|four|five)\b", re.I)
_PARENTHETICAL_SIZE = re.compile(
    r"\(\s*\d+(?:\.\d+)?\s*(?:-\s*)?"
    r"(?:fluid\s+)?(?:ounces?|oz|pounds?|lbs?|grams?|g|kilograms?|kg|"
    r"milliliters?|ml|liters?|l)\s*\)",
    re.IGNORECASE,
)
_UNIT_ALIASES = {
    "fluid ounces": "fluid ounce",
    "fl oz": "fluid ounce",
    "tablespoons": "tablespoon",
    "tbsp": "tablespoon",
    "teaspoons": "teaspoon",
    "tsp": "teaspoon",
    "cups": "cup",
    "ounces": "ounce",
    "oz": "ounce",
    "pounds": "pound",
    "lbs": "pound",
    "lb": "pound",
    "grams": "gram",
    "g": "gram",
    "kilograms": "kilogram",
    "kg": "kilogram",
    "milliliters": "milliliter",
    "ml": "milliliter",
    "liters": "liter",
    "l": "liter",
    "quarts": "quart",
    "qt": "quart",
    "pints": "pint",
    "pt": "pint",
    "gallons": "gallon",
    "gal": "gallon",
    "cloves": "clove",
    "slices": "slice",
    "packages": "package",
    "package": "package",
    "pkg": "package",
    "cans": "can",
    "can": "can",
    "heads": "head",
    "bunches": "bunch",
    "sprigs": "sprig",
    "stalks": "stalk",
    "pieces": "piece",
    "pinches": "pinch",
    "dashes": "dash",
    "sticks": "stick",
    "blocks": "block",
    "block": "block",
}
_UNIT_ALIASES.update({unit: unit for unit in set(_UNIT_ALIASES.values())})
_UNIT_PHRASES = sorted(
    (tuple(unit.split()) for unit in _UNIT_ALIASES),
    key=lambda words: (-len(words), words),
)
_SIZE_MODIFIERS = {"extra", "jumbo", "large", "medium", "small"}
_PREPARATION_MODIFIERS = {
    "beaten",
    "boneless",
    "chopped",
    "coarsely",
    "crushed",
    "cut",
    "diced",
    "divided",
    "drained",
    "finely",
    "freshly",
    "grated",
    "halves",
    "into",
    "lightly",
    "minced",
    "optional",
    "packed",
    "peeled",
    "pitted",
    "rinsed",
    "seeded",
    "shredded",
    "sifted",
    "skinless",
    "sliced",
    "softened",
    "thawed",
    "trimmed",
}


@dataclass(frozen=True)
class ParsedIngredient:
    original_text: str
    quantity: str | None
    quantity_status: str
    unit: str | None
    canonical_name: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class InstructionStepAnalysis:
    count: int | None
    method: str


@dataclass(frozen=True)
class IngredientMatch:
    required_ingredient: str
    available_ingredient: str
    canonical_name: str


@dataclass(frozen=True)
class MatchedRecipe:
    source: str
    recipe_id: str
    title: str | None
    score: float
    matched_ingredients: tuple[IngredientMatch, ...]
    missing_ingredients: tuple[str, ...]
    additional_ingredients: tuple[str, ...]
    instruction_step_count: int | None
    instruction_step_method: str


@dataclass(frozen=True)
class IngredientMatchReport:
    results: tuple[MatchedRecipe, ...]
    total_recipes_in_dataset: int
    candidate_recipes_analyzed: int
    excluded_ingredients: tuple[str, ...]
    max_instruction_steps: int | None
    rejected_excluded_ingredients: int
    rejected_unknown_ingredient_records: int
    rejected_unparseable_ingredient_records: int
    rejected_instruction_step_limit: int
    rejected_unknown_instruction_steps: int
    quantity_sufficiency_verified: bool
    quantity_note: str
    ranking_method: str
    ranking_preferences: tuple[str, ...]


def normalize_ranking_preferences(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError("ranking preferences must be a sequence of preference names")
    normalized_values: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("ranking preferences must contain strings")
        key = normalize_ingredient(value)
        preference = _RANKING_PREFERENCE_ALIASES.get(key)
        if preference is None:
            raise ValueError(f"unsupported ranking preference: {value}")
        if preference not in normalized_values:
            normalized_values.append(preference)
    return tuple(normalized_values)


def _quantity_value(value: str) -> Fraction:
    parts = value.split()
    if len(parts) == 2:
        return Fraction(parts[0]) + Fraction(parts[1])
    return Fraction(parts[0])


def _remove_unit(tokens: list[str]) -> tuple[list[str], str | None]:
    for phrase in _UNIT_PHRASES:
        if tuple(tokens[: len(phrase)]) == phrase:
            unit_key = " ".join(phrase)
            return tokens[len(phrase) :], _UNIT_ALIASES[unit_key]
    return tokens, None


def _canonicalize_tokens(tokens: list[str]) -> str | None:
    while tokens and tokens[0] in _SIZE_MODIFIERS:
        tokens.pop(0)
    remaining = [
        token for token in tokens if token not in _PREPARATION_MODIFIERS
    ]
    canonical = " ".join(remaining).strip()
    if not canonical:
        return None
    return INGREDIENT_ALIASES.get(canonical, canonical)


def parse_ingredient(value: str) -> ParsedIngredient:
    """Parse common leading amounts and units without altering source text."""
    if not isinstance(value, str):
        raise TypeError("ingredient line must be a string")
    return _parse_ingredient_cached(value)


@lru_cache(maxsize=65536)
def _parse_ingredient_cached(value: str) -> ParsedIngredient:
    if not value.strip():
        return ParsedIngredient(value, None, "absent", None, None, ("empty_line",))

    text = unicodedata.normalize("NFKC", value).casefold().replace("⁄", "/")
    text = re.sub(r"\badvertisement\s*$", "", text).strip()
    text = _PARENTHETICAL_SIZE.sub(" ", text)
    text = re.sub(r",?\s*(?:or\s+)?to\s+taste\s*$", "", text)
    text = re.sub(r",?\s*as\s+needed\s*$", "", text)
    warnings: list[str] = []
    quantity: Fraction | None = None
    quantity_status = "absent"
    range_match = _RANGE_AMOUNT.match(text)
    numeric_match = None if range_match else _NUMERIC_AMOUNT.match(text)
    word_match = None if range_match or numeric_match else _WORD_AMOUNT.match(text)

    if range_match:
        quantity_status = "unsupported_range"
        warnings.append("quantity_range_not_compared")
        text = text[range_match.end() :]
    elif numeric_match:
        amount = next(group for group in numeric_match.groupdict().values() if group)
        try:
            quantity = _quantity_value(amount)
            quantity_status = "parsed"
        except (ValueError, ZeroDivisionError):
            quantity_status = "unparsed"
            warnings.append("quantity_not_parsed")
        text = text[numeric_match.end() :]
    elif word_match:
        quantity = _WORD_AMOUNTS[word_match.group("amount").casefold()]
        quantity_status = "parsed"
        text = text[word_match.end() :]

    tokens = normalize_ingredient(text).split()
    unit = None
    if quantity_status != "absent":
        tokens, unit = _remove_unit(tokens)
    tokens = [token for token in tokens if token not in _SIZE_MODIFIERS]
    canonical_name = _canonicalize_tokens(tokens)
    if canonical_name is None:
        warnings.append("ingredient_name_unparsed")

    return ParsedIngredient(
        original_text=value,
        quantity=str(quantity) if quantity is not None else None,
        quantity_status=quantity_status,
        unit=unit,
        canonical_name=canonical_name,
        warnings=tuple(warnings),
    )


_NUMBERED_STEP = re.compile(r"(?<!\S)(?:step\s+)?(\d+)[.)]\s+", re.IGNORECASE)


@lru_cache(maxsize=32768)
def count_instruction_steps(instructions: str | None) -> InstructionStepAnalysis:
    """Count explicit numbered steps or newline-delimited blocks conservatively."""
    if not isinstance(instructions, str) or not instructions.strip():
        return InstructionStepAnalysis(None, "missing_or_blank")

    numbered = [int(match.group(1)) for match in _NUMBERED_STEP.finditer(instructions)]
    if numbered:
        if numbered == list(range(1, len(numbered) + 1)):
            return InstructionStepAnalysis(len(numbered), "numbered_steps")
        return InstructionStepAnalysis(None, "ambiguous_numbering")

    blocks = [line.strip() for line in instructions.splitlines() if line.strip()]
    if len(blocks) > 1:
        if any(len(re.findall(r"[.!?](?:\s+|$)", block)) > 1 for block in blocks):
            return InstructionStepAnalysis(None, "ambiguous_multisentence_blocks")
        return InstructionStepAnalysis(len(blocks), "newline_delimited_blocks")

    sentence_endings = re.findall(r"[.!?](?:\s+|$)", blocks[0])
    if len(sentence_endings) > 1:
        return InstructionStepAnalysis(None, "ambiguous_single_block")
    return InstructionStepAnalysis(1, "single_instruction_block")


def _unique_parsed_ingredients(
    values: Sequence[str], label: str
) -> tuple[tuple[str, ParsedIngredient], ...]:
    parsed: list[tuple[str, ParsedIngredient]] = []
    seen: set[str] = set()
    for value in values:
        ingredient = parse_ingredient(value)
        canonical = ingredient.canonical_name
        if canonical is None:
            raise ValueError(f"{label} ingredient could not be parsed: {value!r}")
        if canonical not in seen:
            parsed.append((value, ingredient))
            seen.add(canonical)
    return tuple(parsed)


def _candidate_query(available: tuple[tuple[str, ParsedIngredient], ...]) -> str:
    expressions: set[str] = set()
    for _, ingredient in available:
        canonical = ingredient.canonical_name
        if canonical is None:
            continue
        variants = {canonical}
        variants.update(
            alias for alias, target in INGREDIENT_ALIASES.items() if target == canonical
        )
        for variant in variants:
            tokens = sorted(set(normalize_ingredient(variant).split()))
            if tokens:
                token_query = " AND ".join(f'"{token}"' for token in tokens)
                expressions.add(f"ingredients : ({token_query})")
    if not expressions:
        raise ValueError("available ingredients produced no searchable terms")
    return " OR ".join(f"({expression})" for expression in sorted(expressions))


def _match_recipe(
    recipe: dict[str, object],
    ingredient_lines: list[tuple[str | None, str | None, str | None]],
    available: tuple[tuple[str, ParsedIngredient], ...],
    excluded_names: set[str],
    max_instruction_steps: int | None,
) -> tuple[MatchedRecipe | None, str | None]:
    parsed_lines = ingredient_lines
    unknown_ingredient = any(canonical is None for _, canonical, _ in parsed_lines)

    if excluded_names and unknown_ingredient:
        return None, "unknown_ingredients"
    for _, name, normalized_text in parsed_lines:
        if name is None:
            continue
        original_tokens = set((normalized_text or "").split())
        if any(
            name == excluded_name
            or set(excluded_name.split()).issubset(original_tokens)
            for excluded_name in excluded_names
        ):
            return None, "excluded_ingredient"

    available_by_name = {
        ingredient.canonical_name: original
        for original, ingredient in available
        if ingredient.canonical_name is not None
    }
    matched: list[IngredientMatch] = []
    missing: list[str] = []
    for original, canonical, _ in parsed_lines:
        if canonical is not None and canonical in available_by_name:
            matched.append(
                IngredientMatch(
                    required_ingredient=original,
                    available_ingredient=available_by_name[canonical],
                    canonical_name=canonical,
                )
            )
        elif original is not None:
            missing.append(original)

    if not matched:
        return None, None

    step_analysis = InstructionStepAnalysis(
        count=(
            recipe["instruction_step_count"]
            if isinstance(recipe["instruction_step_count"], int)
            else None
        ),
        method=str(recipe["instruction_step_method"]),
    )
    if max_instruction_steps is not None:
        if step_analysis.count is None:
            return None, "unknown_instruction_steps"
        if step_analysis.count > max_instruction_steps:
            return None, "instruction_step_limit"

    additional = tuple(
        original
        for original, ingredient in available
        if ingredient.canonical_name not in {
            canonical
            for _, canonical, _ in parsed_lines
            if canonical is not None
        }
    )
    score = len(matched) / len(parsed_lines) if parsed_lines else 0.0
    return (
        MatchedRecipe(
            source=str(recipe["source"]),
            recipe_id=str(recipe["recipe_id"]),
            title=recipe["title"] if isinstance(recipe["title"], str) else None,
            score=score,
            matched_ingredients=tuple(matched),
            missing_ingredients=tuple(missing),
            additional_ingredients=additional,
            instruction_step_count=step_analysis.count,
            instruction_step_method=step_analysis.method,
        ),
        None,
    )


def match_recipes(
    connection: sqlite3.Connection,
    available_ingredients: Sequence[str],
    *,
    excluded_ingredients: Sequence[str] = (),
    max_instruction_steps: int | None = None,
    ranking_preferences: Sequence[str] = (),
    limit: int = 20,
) -> IngredientMatchReport:
    """Rank recipes by matched ingredient-line coverage without relaxing constraints."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    if max_instruction_steps is not None and (
        isinstance(max_instruction_steps, bool)
        or not isinstance(max_instruction_steps, int)
        or max_instruction_steps < 1
    ):
        raise ValueError("maximum instruction steps must be a positive integer")
    if isinstance(available_ingredients, str) or isinstance(excluded_ingredients, str):
        raise TypeError("ingredient inputs must be sequences of ingredient strings")
    preferences = normalize_ranking_preferences(ranking_preferences)

    available = _unique_parsed_ingredients(
        available_ingredients, "available"
    )
    if not available:
        raise ValueError("at least one available ingredient is required")
    excluded = _unique_parsed_ingredients(excluded_ingredients, "excluded")
    excluded_names = {
        ingredient.canonical_name
        for _, ingredient in excluded
        if ingredient.canonical_name is not None
    }

    total_recipes = connection.execute(
        "SELECT COUNT(*) FROM recipes"
    ).fetchone()[0]
    rejected_unknown_ingredients = connection.execute(
        """
        SELECT COUNT(*)
        FROM recipes
        WHERE ingredients_valid = 0
           OR NOT EXISTS (
               SELECT 1
               FROM recipe_ingredients
               WHERE recipe_ingredients.source = recipes.source
                 AND recipe_ingredients.recipe_id = recipes.recipe_id
           )
        """
    ).fetchone()[0]
    candidate_query = _candidate_query(available)
    rows = connection.execute(
        """
        SELECT
            recipes.source,
            recipes.recipe_id,
            recipes.title,
            recipes.ingredients_valid,
            recipe_ingredients.original_text,
            recipe_ingredients.normalized_text,
            recipe_ingredients.canonical_name,
            recipe_ingredients.position,
            recipes.instruction_step_count,
            recipes.instruction_step_method
        FROM recipe_fts
        JOIN recipes
          ON recipes.source = recipe_fts.source
         AND recipes.recipe_id = recipe_fts.recipe_id
        LEFT JOIN recipe_ingredients
          ON recipe_ingredients.source = recipes.source
         AND recipe_ingredients.recipe_id = recipes.recipe_id
        WHERE recipe_fts MATCH ? AND recipes.ingredients_valid = 1
        ORDER BY recipes.source, recipes.recipe_id, recipe_ingredients.position
        """,
        (candidate_query,),
    )
    results: list[MatchedRecipe] = []
    candidate_recipes = 0
    rejected_excluded = 0
    rejected_unparseable_ingredients = 0
    rejected_step_limit = 0
    rejected_unknown_steps = 0
    current_key: tuple[str, str] | None = None
    current_recipe: dict[str, object] | None = None
    current_lines: list[tuple[str | None, str | None, str | None]] = []

    def finish_recipe() -> None:
        nonlocal rejected_excluded
        nonlocal rejected_unparseable_ingredients
        nonlocal rejected_step_limit
        nonlocal rejected_unknown_steps
        if current_recipe is None:
            return
        if not current_lines:
            rejected_unparseable_ingredients += 1
            return
        result, reason = _match_recipe(
            current_recipe,
            current_lines,
            available,
            excluded_names,
            max_instruction_steps,
        )
        if reason == "excluded_ingredient":
            rejected_excluded += 1
        elif reason == "unknown_ingredients":
            rejected_unparseable_ingredients += 1
        elif reason == "instruction_step_limit":
            rejected_step_limit += 1
        elif reason == "unknown_instruction_steps":
            rejected_unknown_steps += 1
        elif result is not None:
            results.append(result)

    for row in rows:
        key = (row["source"], row["recipe_id"])
        if key != current_key:
            finish_recipe()
            candidate_recipes += 1
            current_key = key
            current_recipe = {
                "source": row["source"],
                "recipe_id": row["recipe_id"],
                "title": row["title"],
                "ingredients_valid": row["ingredients_valid"],
                "instruction_step_count": row["instruction_step_count"],
                "instruction_step_method": row["instruction_step_method"],
            }
            current_lines = []
        if row["position"] is not None:
            current_lines.append(
                (
                    row["original_text"],
                    row["canonical_name"],
                    row["normalized_text"],
                )
            )
    finish_recipe()

    def ranking_key(result: MatchedRecipe) -> tuple[object, ...]:
        preference_key: list[object] = []
        for preference in preferences:
            if preference == "fewer_missing":
                preference_key.append(len(result.missing_ingredients))
            elif preference == "fewer_additional":
                preference_key.append(len(result.additional_ingredients))
            else:
                preference_key.append(
                    result.instruction_step_count
                    if result.instruction_step_count is not None
                    else float("inf")
                )
        return (
            *preference_key,
            -result.score,
            -len(result.matched_ingredients),
            len(result.missing_ingredients),
            len(result.additional_ingredients),
            (
                result.instruction_step_count
                if result.instruction_step_count is not None
                else float("inf")
            ),
            result.source,
            result.recipe_id,
        )
    results.sort(key=ranking_key)
    preference_description = (
        "Apply preferences in the supplied order ("
        + ", ".join(preferences)
        + "), then sort by "
        if preferences
        else "Sort by "
    )
    return IngredientMatchReport(
        results=tuple(results[:limit]),
        total_recipes_in_dataset=total_recipes,
        candidate_recipes_analyzed=candidate_recipes,
        excluded_ingredients=tuple(value for value, _ in excluded),
        max_instruction_steps=max_instruction_steps,
        rejected_excluded_ingredients=rejected_excluded,
        rejected_unknown_ingredient_records=rejected_unknown_ingredients,
        rejected_unparseable_ingredient_records=rejected_unparseable_ingredients,
        rejected_instruction_step_limit=rejected_step_limit,
        rejected_unknown_instruction_steps=rejected_unknown_steps,
        quantity_sufficiency_verified=False,
        quantity_note=(
            "Ingredient quantities are parsed when recognizable but are not "
            "compared; having an ingredient does not verify sufficient quantity."
        ),
        ranking_method=(
            preference_description
            + "matched ingredient-line coverage (matched lines / required "
            "lines), then matched line count, fewer missing lines, fewer additional "
            "available ingredients, fewer instruction steps, source, and recipe ID."
        ),
        ranking_preferences=preferences,
    )
