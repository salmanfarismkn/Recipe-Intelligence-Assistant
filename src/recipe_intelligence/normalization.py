"""Conservative text normalization for ingredient retrieval."""

from __future__ import annotations

import unicodedata


def normalize_ingredient(value: str) -> str:
    """Normalize Unicode, case, punctuation, and whitespace without parsing meaning."""
    if not isinstance(value, str):
        raise TypeError("ingredient text must be a string")

    normalized = unicodedata.normalize("NFKC", value).casefold()
    characters = (
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return " ".join("".join(characters).split())
