# Recipe Intelligence Assistant

A local recipe search and evidence-grounded cooking assistant built on the
Eight Portions recipe dataset. The application preserves source recipe records,
uses deterministic SQLite-backed search, matching, and analytics, and exposes
them through FastAPI and a lightweight browser chat.

## Requirements

- Python 3.10 or newer
- The Eight Portions source archive (downloaded by the project script)
- Optional: a Gemini API key and model for natural-language interpretation

The deterministic application runs without Gemini credentials. There is no
Docker configuration in this repository.

The pinned ZIP is about 53.4 MB; a full ingested SQLite database was about
1.26 GB in the verified run. Allow at least 2 GiB of free space where the
database is written for the database and SQLite ingestion/indexing headroom.
This is a practical recommendation, not a platform-independent minimum.

## Run locally

From the repository root in PowerShell:

```powershell
python -m pip install -e ".[test]"
python scripts\download_dataset.py
python scripts\ingest_dataset.py data\raw\recipes_raw.zip
uvicorn recipe_intelligence.api:app --reload
```

Open <http://127.0.0.1:8000/> for the web chat, `/docs` for interactive API
documentation, and `/health` for database/provider status. The default database
is `data/processed/recipes.sqlite3`; set `RECIPE_DATABASE_PATH` to use another
path. Dataset archives and generated databases are ignored by Git.

To run tests:

```powershell
python -m pytest -q
```

## Gemini (optional)

Set `LLM_ENABLED=true`, `GEMINI_API_KEY`, and `GEMINI_MODEL` in the server
environment to enable the existing Gemini provider. Keep credentials out of
source files and browser code. Gemini can interpret and frame requests but
does not calculate statistics, satisfy hard constraints, or supply recipe
records. The API remains usable without the provider.

## Project documentation

- [API and conversation behavior](docs/api-and-conversations.md)
- [Architecture](docs/architecture.md)
- [Design decisions and limitations](docs/design-decisions.md)
- [Dataset inventory and ingestion assumptions](docs/dataset-inventory.md)
- [Gemini integration](docs/gemini-integration.md)
- [Substitution evidence model](docs/substitution-engine.md)
- [Release evaluation and reproducibility](docs/evaluation.md)

## Dataset and scope

The application uses a pinned snapshot of three source-specific JSON maps with
125,164 recipe IDs. Ingestion retains the filename-derived source code, stable
ID, original record, and ordered original ingredient strings; normalized values
are stored separately. See the [dataset inventory](docs/dataset-inventory.md)
and machine-readable [inventory report](reports/dataset_inventory.json).

Search is deterministic keyword retrieval (FTS5), not semantic search.
Ingredient matching reports quantity sufficiency as unverified. Substitution
results are classified from local recipe comparisons and may refuse to
recommend an adaptation. Ingredient-text restrictions are not allergy-safety
assurances. See the [evaluation report](reports/evaluation_report.json) for
measured full-dataset results and unresolved verification items.
