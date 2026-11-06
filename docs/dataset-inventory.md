# Eight Portions dataset inventory

## Source snapshot

- Official dataset page: <https://eightportions.com/datasets/Recipes/>
- Archive: <https://eightportions.com/recipes_raw.zip>
- Archive SHA-256: `1ae20f3260313d501edf23c22c6f1875e87917f5a8e2ff13b40450719322c81b`
- Archive size observed: 53,355,492 bytes
- Database license included in the archive: Open Data Commons Attribution License (ODC-By)
- The publisher says source URLs in the hosted data are hashed; the JSON records do not include an explicit source-site field.

The archive contains three JSON objects. Each top-level key is a stable recipe ID, and its value is a recipe object. Source attribution for ingestion must therefore be retained from the filename rather than inferred from a recipe record.

| File | Source code | Records |
|---|---:|---:|
| `recipes_raw_nosource_ar.json` | `ar` | 39,802 |
| `recipes_raw_nosource_epi.json` | `epi` | 25,323 |
| `recipes_raw_nosource_fn.json` | `fn` | 60,039 |
| **Total** | | **125,164** |

Across this snapshot, no duplicate JSON object keys or cross-file recipe-ID collisions were found. There are 125,164 distinct IDs. Ingestion must still check for both duplicate keys and cross-source ID collisions; keys must not be silently overwritten. A collision across source files is retained as separate records identified by `(source, recipe_id)` and reported. The inventory includes expected and actual per-source/total record counts, so snapshot drift is visible.

## Observed schema and data quality

The four expected fields are `title`, `ingredients`, `instructions`, and `picture_link`. Present ingredient values are lists of strings in the inspected snapshot. Titles and instructions are usually text, but can be null or blank. Picture links can be null. The inspection records missing and null values separately.

| File | Empty records missing all four fields | Null title | Null instructions | Null picture link | Empty ingredient lists | Blank instructions |
|---|---:|---:|---:|---:|---:|---:|
| AR | 280 | 0 | 0 | 0 | 0 | 0 |
| Epi | 0 | 0 | 0 | 13,204 | 47 | 23 |
| FN | 237 | 52 | 174 | 28,850 | 1,629 | 0 |

The 517 empty records are present in the source files and are part of the reported record totals. They must be preserved and flagged, not filtered out. The original source strings, including whitespace and nulls, must remain distinct from any normalized or interpreted values.

The SQLite store uses `(source, recipe_id)` as its primary key. It keeps the original record JSON and each ingredient value in order, while a separate ingredient table stores normalized text and conservative parsed fields (canonical ingredient identity, quantity status, recognized unit, and parser warnings). Recipe rows also store the instruction-step count and the method used to derive it. These derived fields never replace or alter source content. Import is transactional and replaces the previous snapshot only after the new archive has been inspected; repeating an import does not duplicate records. The FTS5 index supports title-only, ingredient-only, and combined keyword retrieval. Queries require all normalized terms and rank deterministically with BM25. This is keyword full-text retrieval, not semantic search.

`RecipeFilters` are applied before search limits and are shared with statistics. Ingredient include/exclude filters use exact contiguous normalized text, without aliases or stemming. Exclusion filtering also omits recipes whose ingredient list is missing or malformed, rather than treating unknown ingredients as safe. This text matching is not a medical or allergy-safety guarantee.

The current deterministic statistics are recipe count, average ingredient-line count, and ingredient prevalence. Results include the calculation method, numerator, denominator, filters, total analyzed recipes, and records excluded for unusable ingredient lists. Average and prevalence denominators include recipes with a valid ingredient list (including an empty list); missing or malformed lists are counted as excluded.

Conversation state is persisted by explicit session ID and contains the active recipe, filters, available ingredients, preferences, previous result IDs, and current task. A new ID loads an empty state; restrictions merge into its filters and make recipe search the active task.

Ingredient matching uses explicit aliases and ranks candidates by matched ingredient lines divided by the recipe's parsed ingredient-line count. It does not compare quantities, so every result reports that quantity sufficiency was not verified. An ingredient exclusion is a hard constraint: a recipe with an unparseable ingredient line is rejected when exclusions are active. A maximum-step constraint also rejects recipes whose instruction-step count is ambiguous or unavailable; it is never relaxed.

## Ingestion assumptions

1. Treat each source file as a mapping from its source-specific recipe IDs to original recipe objects.
2. Keep the filename-derived source code and ID; use `(source, recipe_id)` as the persistence key. Do not merge files with a plain dictionary update.
3. Keep the source archive unchanged. Store original field values separately from validation results and normalized ingredients.
4. Report missing fields, explicit nulls, empty ingredient lists, blank text, invalid record shapes, duplicate JSON keys, and cross-source ID collisions. Do not silently drop records.
5. Before ingestion, compare the archive SHA-256 with this pinned snapshot. Review any changed source archive and refresh expected counts only after inspecting it.
6. The supplied archive contains a license for the database; preserve attribution and consult the license for redistribution.

## Reproducing the inventory

From the repository root:

```powershell
python -m pip install -e ".[test]"
python scripts\download_dataset.py
python -m recipe_intelligence.ingestion data\raw\recipes_raw.zip --output data\raw\inventory.json
python scripts\ingest_dataset.py data\raw\recipes_raw.zip
pytest
```

The download command writes the verified archive locally; generated dataset files and inventory reports are not part of the source tree by default.
