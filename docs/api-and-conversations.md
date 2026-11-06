# API and conversational sessions

The application exposes the existing deterministic recipe services over
FastAPI, provides persisted conversational state, and serves a lightweight
responsive browser interface from the same origin.

## Run locally

Install the application and test dependencies, then initialize the local SQLite
database with the source archive:

```powershell
python -m pip install -e ".[test]"
python scripts\download_dataset.py
python scripts\ingest_dataset.py data\raw\recipes_raw.zip
uvicorn recipe_intelligence.api:app --reload
```

Open `http://127.0.0.1:8000/` for the web chat. The frontend and versioned API
share the same origin; no separate frontend server, build step, CORS rule, or
frontend environment variable is required. `/docs` and `/openapi.json` remain
available for API exploration.

By default the database path is `data/processed/recipes.sqlite3`. Override it
with `RECIPE_DATABASE_PATH`. FastAPI's OpenAPI schema and interactive docs are
available at `/openapi.json` and `/docs`.

Gemini remains optional and disabled by default. To enable natural-language
interpretation and constrained response framing, set `LLM_ENABLED=true`,
`GEMINI_API_KEY`, and `GEMINI_MODEL` as described in
[Gemini integration](./gemini-integration.md). Health remains `ok` when Gemini
is disabled; a configured provider's availability is reported separately.
Deterministic endpoints do not require Gemini.

## Endpoints

### Health

`GET /health`

```json
{"status":"ok","database":"ok","gemini":"disabled"}
```

The database is checked with a local query. Gemini is reported as `configured`
or `disabled`; this endpoint makes no paid provider request.

### Search and details

`POST /api/v1/recipes/search`

```json
{
  "query": "garlic bread",
  "field": "title",
  "filters": {"exclude_ingredients": ["peanut"]},
  "limit": 10,
  "include_details": true
}
```

Instead of `query`, callers may send `natural_language` (up to 2,000
characters). With Gemini enabled, the existing typed interpreter validates
that operation. With Gemini disabled or unavailable, a bounded deterministic
parser supports simple `find`, `search for`, and `show me` recipe searches.
Unsupported ambiguous natural language is rejected or unavailable; it is not
guessed. Search results include source, stable recipe ID, a
`source:recipe-id` reference, title, retrieval rank, applied filters, and result
count. Original ingredient strings and instructions are included only when
requested.

Schematic response shape (placeholder values are not dataset facts):

```json
{
  "query": "garlic bread",
  "field": "title",
  "filters_applied": {
    "sources": [],
    "title_query": null,
    "include_ingredients": [],
    "exclude_ingredients": ["peanut"]
  },
  "result_count": 1,
  "results": [
    {
      "source": "ar",
      "recipe_id": "<stable-id>",
      "reference": "ar:<stable-id>",
      "title": "<original dataset title>",
      "rank": "<retrieval rank>"
    }
  ]
}
```

`GET /api/v1/recipes/{recipe_id}?source=ar` returns the preserved original
record. `source` is optional when the ID is unique; duplicate IDs across
sources require it. The path accepts URL-encoded stable IDs containing `/`.
Unknown records return `404`.
The detail response includes `source`, `recipe_id`, `reference`, the complete
original `recipe` object, and `validation_issues`.

### Analytics

`POST /api/v1/analytics`

```json
{
  "statistic": "ingredient_prevalence",
  "ingredient": "garlic",
  "filters": {"sources": ["ar"]}
}
```

Supported values are `recipe_count`, `average_ingredient_lines`, and
`ingredient_prevalence`. The existing statistics service computes each result
programmatically over the full filtered set, including denominator, analyzed
and excluded recipe counts, filters, and method. Gemini is not used.
The response is `{"result": <StatisticResult>}`; the nested result contains
`statistic`, `value`, `unit`, `numerator`, `denominator`, `filters`,
`analyzed_recipe_count`, `excluded_recipe_count`, and `calculation_method`.

### Ingredient matching

`POST /api/v1/matching`

```json
{
  "available_ingredients": ["garlic", "bread"],
  "excluded_ingredients": ["peanut"],
  "max_instruction_steps": 5,
  "ranking_preferences": ["fewer_additional", "fewer_steps"],
  "limit": 20
}
```

Available ingredient strings may include quantities; quantity sufficiency is
still reported as unverified. Preferences are `fewer_missing`,
`fewer_additional`, and `fewer_steps`. They only reorder recipes that already
pass hard exclusions, ingredient validity, and any maximum-step constraint.
When multiple preferences conflict, they are applied lexicographically in
request order; the normal deterministic score, matched-line count, and stable
source/ID tie-breaks follow. The report includes ranking method, score
explanation, eligibility constraint details, matched and missing ingredients,
additional available ingredients, and quantity-verification status.
Responses use `{"report": <IngredientMatchReport>}`. Each ranked result includes
its score, matched and missing ingredient lines, additional available
ingredients, and step-count method.

### Substitution

`POST /api/v1/substitutions`

```json
{
  "recipe_id": "original-recipe-id",
  "source": "ar",
  "ingredient_to_replace": "cream",
  "replacement_ingredients": ["oat cream"],
  "excluded_ingredients": ["peanut"],
  "dietary_restrictions": ["dairy-free"]
}
```

The API resolves the actual recipe and returns the substitution service's
evidence classifications, IDs, restrictions, and refusal decisions unchanged.
An ambiguous recipe ID returns `409`; an unknown ID returns `404`. Text-based
restrictions are not medical or allergy-safety assurances.
The response wraps the existing proposal as `{"proposal": <SubstitutionProposal>}`;
its options retain the evidence class, supporting source/recipe IDs, rejection
reason, and any revised ingredient list.

## Sessions

- `POST /api/v1/sessions` creates a UUID session and returns its initial state.
- `POST /api/v1/sessions/{session_id}/messages` interprets one message, executes
  the relevant service, then atomically persists valid state changes.
- `GET /api/v1/sessions/{session_id}` retrieves structured state and last-update
  metadata.
- `DELETE /api/v1/sessions/{session_id}` explicitly removes a session.

Session creation returns `session_id`, `created_at`, and `state`. Message
responses include `status`, deterministic or grounded `response`, operation
`result` when available, `used_gemini`, and the resulting structured `state`.

Session state persists in SQLite's existing `conversation_sessions` table. It
contains active task and recipe, previous verified result IDs, search filters
and context, available ingredients, exclusions, maximum steps, ordered ranking
preferences, and substitution context. Added state fields have backward-
compatible defaults when older state JSON is loaded. A session is not created
implicitly by a message request; unknown sessions return `404`. IDs must be
UUIDs.

Within one application process, messages for a given session are serialized
with a per-session lock. State is committed transactionally, and interpretation
or service failures do not persist a partial update. SQLite busy timeout is
five seconds. This is a local SQLite application, not a distributed
multi-worker session coordinator; use a single worker when concurrent writers
must have strict per-session ordering.

There is no automatic expiration. Sessions persist until explicitly deleted or
removed by local database maintenance.

### Browser interface and local session history

The root page serves the vanilla HTML, CSS, and JavaScript client. It uses the
same session, recipe detail, search, analytics, matching, and substitution
endpoints documented above. Conversational messages go through the session
message endpoint; focused tools call their deterministic endpoint directly and
are labeled as not changing conversation context.

The API does not expose a session-list endpoint. Accordingly, the browser
sidebar does not claim to be a server-side session history: this UI records only
session IDs and exchanges made through that browser in local storage, then
revalidates each saved session against the API on startup. Message content and
response payloads stay in that browser's local history; structured state lives
in SQLite and is obtained from the session retrieval endpoint. Deleting a
conversation invokes the API's explicit delete operation and removes its local
entry. Clearing browser storage removes the local transcript/session index,
but does not delete SQLite sessions.

Recipe and model-provided text is inserted as text, not interpreted as HTML.
When the API or Gemini is unavailable, the client displays an error or the
backend's deterministic fallback response rather than presenting stale results
as current. The UI contains no Gemini credentials and loads no third-party
frontend assets.

### Context and follow-ups

Verified result references resolve by ordinal only against the previous result
IDs. For example:

1. `Find garlic bread`
2. `Show the second recipe.`
3. `Show its original instructions.`
4. `Exclude egg too.`

Selecting a previous result verifies it against the current database before
setting it as the active recipe. “Show the second recipe” returns its preserved
ingredient and instruction strings; “Show its original instructions” returns
the source instruction field. A restriction follow-up updates and reruns the
active search/match, preserving existing same-task restrictions. A new task
does not inherit unrelated search filters, matching exclusions, or ranking
preferences. Supported deterministic follow-ups include excluding/removing an
ingredient, selecting a prior result, showing its source instructions,
preferring fewer missing/additional ingredients or fewer steps, and replacing
an ingredient in a uniquely resolved active recipe. `Show only recipes with
fewer steps` converts the previous numeric result ceiling to a stricter ceiling;
unknown step counts are never treated as verified.

Without Gemini, the session layer supports these explicit follow-ups and simple
searches or ingredient lists starting with `I have`. Other free-form requests
return a clarification rather than an invented operation. When Gemini is
enabled, it interprets requests through the Stage 6 schema; application code
still resolves recipe IDs, merges restrictions, executes deterministic
services, validates output, and renders evidence.

## Errors and limits

Errors use a stable envelope:

```json
{"error":{"code":"recipe_not_found","message":"Recipe not found."}}
```

Expected statuses include `400` invalid session ID, `404` missing recipe/session,
`409` ambiguous recipe ID, `413` body over 64 KiB, `422` invalid parameters or
operation, and `503` database or required interpretation unavailability.
Bodies are limited to 64 KiB, and individual strings and collections have
schema-level limits. Credentials and raw exception traces are not returned.

The current conversational fallback is intentionally limited; it is not a
general natural-language parser. Per-session locks are process-local, and
session records do not expire automatically.
