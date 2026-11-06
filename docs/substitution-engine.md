# Evidence-based substitution engine

The local substitution service is exposed by
`recipe_intelligence.substitution.propose_substitutions`. It accepts a recipe
source and stable ID, an ingredient to replace, optional proposed replacements,
hard exclusions, dietary restrictions, and ingredient-name preferences. A missing
recipe returns `None`; an ingredient that is absent or ambiguous in the requested
recipe raises `ValueError`.

The service reads the preserved original recipe through the existing retrieval
function and uses the stored `recipe_ingredients.canonical_name`,
`recipe_ingredients.original_text`, and recipe instruction fields. Candidate
lookups use canonical-ingredient indexes; the request does not reparse every
dataset record. Discovered candidates come from recipes containing the replaced
ingredient. The engine then checks whether each candidate occurs in comparable
recipes that omit the replaced ingredient. Co-occurring recipes are not treated
as supporting substitutions. A caller may also ask to evaluate specific
replacements, including ingredients absent from the dataset.

## Evidence rules

An evidence record identifies its actual `(source, recipe_id)`, original
replacement ingredient line(s), shared ingredients, shared title terms,
recognized preparation methods, and relevant instruction excerpts. Recipes that
contain both the old and proposed ingredient are listed separately as
co-occurrence IDs and are never treated as supporting substitutions.

Preparation methods are a small deterministic verb lexicon applied to source
instructions. Title matching is a normalized token overlap with generic title
words removed. These are retrieval signals, not an asserted dish taxonomy.

| Classification | Rule |
|---|---|
| **Direct** | A comparison recipe omits the old ingredient and contains the proposed ingredient; both recipes have fully parsed ingredient lists; at least two other canonical ingredients are shared, covering at least 75% of the original recipe's other parsed ingredients; a preparation method is shared; and either the known instruction-step counts match or a title term is shared. |
| **Indirect** | Such a comparison shares at least two other ingredients covering at least 50%, plus a preparation method or title term. |
| **Weak** | The proposed ingredient occurs in a recipe that omits the old ingredient, but no comparison meets the indirect rule. |
| **No evidence** | No usable comparison record for the proposed ingredient was found. |

These categories describe the retrieved dataset evidence; even the direct class
does not prove interchangeability or an equivalent quantity. A recommendation is
considered sufficiently supported only for a direct, non-conflicting option that
passes all hard restrictions. If multiple options meet the direct rule, they are
reported as competing evidence and none is selected automatically.

The role labels (binding, moisture, fat, thickening, flavor, texture, and
structure) come from a small ingredient-name lexicon. Each is explicitly marked
heuristic and does not influence evidence strength.

## Adaptation and restrictions

For a sufficiently supported option, the revised ingredient list preserves
other source ingredient strings exactly and replaces only the selected source
line with the replacement's canonical name. No quantity is transferred. Source
instructions are not rewritten: lines naming the replaced ingredient are
returned for manual adaptation because the dataset does not establish an exact
safe instruction rewrite. Taste, texture, and preparation trade-offs are marked
unknown unless the local records actually establish them (this version does not
infer them from co-occurrence).

Ingredient exclusions and supported `<ingredient>-free` restrictions are checked
against the complete proposed ingredient list. `dairy-free` expands to an
explicit dairy-name list; recognized plant milks (oat, almond, soy, rice,
coconut, and cashew) are not treated as dairy milk. Any unparseable ingredient
line prevents verification when a hard restriction is active. Unsupported
restriction forms raise an error instead of being silently ignored.

`milk-free` is intentionally stricter than `dairy-free`: it blocks plant milk
as well. An explicit milk exclusion also takes precedence over the plant-milk
allowance used by the dairy-free rule.

These checks inspect recipe text only. They are not medical or allergy-safety
guarantees and cannot detect undeclared ingredients or cross-contamination.
Preferences are parsed as preferred replacement ingredient names and only
influence ordering after hard constraints and support status; they cannot
upgrade weak evidence.

## Limitations

- The dataset has no explicit substitution annotations or validated ingredient
  roles. Similar composition and method are evidence of context, not a controlled
  substitution experiment.
- Preparation-method recognition and title overlap are deliberately small
  heuristics. Missing or unusual wording can lower evidence strength.
- Quantity sufficiency and unit equivalence are not evaluated.
- Ambiguous, repeated, or malformed target ingredient identification is refused.
- A replacement without adequate local support is returned as weak/no-evidence,
  not as a reliable adaptation.

Run the substitution tests with:

```powershell
python -m pytest tests\test_substitution.py -q
```
