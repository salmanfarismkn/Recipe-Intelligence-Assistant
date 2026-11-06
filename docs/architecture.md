# Architecture

## Shape

The project is a modular monolith: one Python package, one FastAPI process,
one SQLite database, and static browser assets served from the same origin.
There are no queues, vector databases, distributed services, or frontend build
pipeline.

```text
Browser (static HTML/CSS/JavaScript)
                 |
                 v
FastAPI routes and session orchestration
       |         |           |
       v         v           v
 SQLite       Gemini      deterministic
 persistence  provider    services
                           search / stats /
                           matching / substitution
```

## Modules and boundaries

| Module | Responsibility |
|---|---|
| `ingestion.py` | Inspect the ZIP/JSON inputs, enforce schema assumptions, report inventory and source counts. |
| `database.py` | SQLite schema and migrations, transactional ingestion, preserved source records, normalized ingredient rows, and FTS5 indexing. |
| `normalization.py` | Shared deterministic text normalization. |
| `filters.py` | Validated, parameterized recipe-filter predicates shared by search and analytics. |
| `retrieval.py` | Title/ingredient/all-field FTS search and exact source-and-ID recipe retrieval. |
| `matching.py` | Conservative ingredient parsing, hard eligibility checks, and deterministic preference-aware ranking. |
| `statistics.py` | Dataset-wide programmatic counts, averages, and ingredient prevalence. |
| `substitution.py` | Local comparison retrieval, evidence classifications, hard restriction checks, and conservative adaptation proposals. |
| `conversation.py` | Typed structured session state and SQLite load/save behavior. |
| `assistant.py` | Typed request validation, deterministic execution, fact assembly, and constrained response rendering. |
| `llm.py` | Optional bounded Gemini structured-JSON provider with explicit failure types. |
| `api.py` | HTTP validation, endpoint orchestration, session reference resolution, per-session locking, and static-file serving. |
| `static/` | Accessible vanilla browser UI; recipe and response strings are rendered as text. |

These are responsibilities within one deployable application, not separate
services. Request handlers call existing domain services rather than duplicating
their calculations.

## Persistence

SQLite stores source recipe rows keyed by `(source, recipe_id)`, original record
JSON, validation issues, instruction-step metadata, ordered source ingredient
values, separate parsed/normalized ingredient fields, and the FTS5 index.
Conversation state is serialized to `conversation_sessions` with explicit
fields for active task/recipe, previous result IDs, search filters, available
ingredients, exclusions, step ceilings, ordered preferences, and substitution
context. State updates use SQLite transactions.

The three source files are separate ID namespaces. Source code is derived from
the filename; records are not merged by plain recipe-ID dictionary updates.
Exact duplicate record content may occur under different IDs; those records
are retained.

## Request flows

### Direct API tasks

1. Validate request shape and limits through Pydantic.
2. Resolve requested filters and source-qualified recipe identity.
3. Invoke the deterministic retrieval, analytics, matching, or substitution
   service.
4. Return the computed result and its applied filters, constraints, denominator,
   or source evidence as appropriate.

### Conversation messages

1. Validate the session UUID and retrieve its persisted structured state.
2. Serialize messages for that session within the process.
3. Resolve explicit ordinal references against actual previous result IDs.
4. Parse or interpret the request; validate extracted entities and allowed
   operations.
5. Execute deterministic services and assemble verified result data.
6. Ask Gemini to select only from supplied fact identifiers when available;
   otherwise return deterministic wording.
7. Persist the new structured state only after successful interpretation and
   service execution.

Clarifications and failed interpretations do not become authoritative state.
New unrelated tasks start without carrying over unrelated filters. Restrictions
and ranking preferences are retained for contextual refinements of the active
task. See [API and conversation behavior](./api-and-conversations.md) for
contracts and examples.

## Failure and deployment boundaries

Input errors use stable HTTP error envelopes; the API does not expose raw
traces or provider credentials. Database connections are opened per request and
closed through FastAPI dependency cleanup. Gemini is optional and its failure
does not disable deterministic services.

Per-session locks are process-local. SQLite busy timeout handles brief local
contention, but the application is not a distributed multi-worker session
coordinator. Session expiration is not implemented. No Docker or production
deployment configuration is supplied.
