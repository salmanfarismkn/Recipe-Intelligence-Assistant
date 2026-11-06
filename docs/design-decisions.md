# Design decisions and limitations

## SQLite and FTS5 instead of external services

The dataset and workload fit a local relational store. SQLite provides
repeatable ingestion, transaction boundaries, stable source-qualified IDs, and
FTS5 keyword retrieval without a separate database service. Retrieval requires
all normalized query terms and ranks results deterministically. It is not
semantic retrieval; embeddings were not added because no comparative retrieval
evaluation justified the added model/storage complexity.

## Preserve source records; derive separately

Original record JSON and each original ingredient value are stored unchanged.
Normalization, parsed quantities, canonical ingredient names, and step counts
are separate derived fields. Ingestion keys records by source and stable ID,
validates inputs before replacing the current snapshot, and records an
inventory. It does not silently drop records with missing fields.

## Ingredient matching

Parsing is intentionally conservative and uses explicit aliases. Hard
exclusions and a maximum-step ceiling determine eligibility before preferences
are applied. Preferences (`fewer_missing`, `fewer_additional`, `fewer_steps`)
are applied lexicographically in request order; deterministic coverage and
source/ID tie-breaks remain. Preferences cannot make an ineligible recipe
eligible. Quantities are not compared, so the result explicitly says quantity
sufficiency is not verified.

## Statistics

Statistics execute against the complete filtered recipe set; they are not
delegated to Gemini. Recipe count includes matching records. Average ingredient
lines and prevalence include records with valid ingredient lists (including an
empty list) and report the valid-list denominator and excluded count. Current
statistics are recipe count, average ingredient-line count, and ingredient
prevalence; median counts and arbitrary combinations are not separate
statistics.

Prevalence uses normalized text-token containment, without aliasing or
stemming. It is deterministic but not a curated ingredient taxonomy.

## Substitution evidence

The engine compares actual local recipes and reports source-qualified evidence
references. Co-occurrence is not treated as proof of replacement. Only direct,
non-conflicting evidence that satisfies all hard restrictions can yield a
revised ingredient list; weaker or conflicting evidence is reported with
uncertainty/refusal. Replacement quantities are not inferred and original
instructions are not silently rewritten. Ingredient roles are heuristic
labels, not verified culinary science.

Dietary and excluded-ingredient checks operate on recorded ingredient text.
`dairy-free` recognizes a defined list and permits recognized plant milks;
`milk-free` explicitly excludes plant milk too. Neither behavior establishes
medical/allergy safety, hidden ingredients, or cross-contamination status.

## Gemini boundary

The optional Gemini provider classifies user intent and can select among
application-created fact IDs. Statistics, recipe identity, hard-constraint
eligibility, evidence classification, and persisted session state remain
application-owned. Malformed or unknown response facts fall back to
deterministic rendering. The provider makes bounded synchronous requests and
is not required to run the application.

No live Gemini result was obtained for this repository snapshot because
credentials were not configured. Mocked provider tests are not evidence of
live availability or model quality.

## Conversation scope

The API persists typed task context rather than treating raw transcript text as
authoritative. Ordinal references are resolved against actual prior result IDs;
ambiguous references ask a clarification. Session updates are serialized within
one process and transactionally saved. There is no automatic expiration or
distributed locking.

Without Gemini, natural-language support is intentionally bounded to explicit
patterns and contextual follow-ups; unsupported phrasing may require a more
direct request. This is not a general conversational parser.

## User interface and deployment

The browser client is vanilla JavaScript served by FastAPI. It stores a local
transcript/session index in browser storage because the API does not provide a
session-list endpoint. Structured session state is authoritative in SQLite.
Dataset strings are inserted as text, not HTML. No API credentials are placed
in the frontend.

No Dockerfile, Compose file, authentication layer, or production multi-worker
configuration is present. Local single-process operation is the verified
deployment target.
