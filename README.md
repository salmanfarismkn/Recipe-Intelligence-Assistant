# Recipe Intelligence Assistant

An evidence-grounded conversational recipe assistant built on the
[Eight Portions Recipe Box dataset](https://eightportions.com/datasets/Recipes/).

The application supports recipe search, dataset analytics, ingredient-based
matching, and evidence-based ingredient substitution across **125,164 recipes**
from three source datasets.

It uses deterministic Python services and SQLite for factual retrieval,
filtering, calculations, and ranking, with optional Google Gemini integration
for natural-language interpretation and response framing.

## Features

- **Recipe search:** Search recipes by title and ingredients, with filters and
  contextual follow-up restrictions.
- **Dataset analytics:** Calculate recipe counts, average ingredient-line
  counts, and ingredient prevalence programmatically.
- **Ingredient matching:** Rank recipes using available ingredients, hard
  exclusions, step limits, and explainable preferences.
- **Ingredient substitution:** Compare local recipe evidence, classify evidence
  strength, and report uncertainty.
- **Conversational interface:** Maintain structured session context across
  multiple messages through a lightweight web interface.

## Technology stack

- Python 3.10+
- FastAPI and Uvicorn
- SQLite and FTS5
- HTML, CSS, and JavaScript
- Google Gemini API (optional at runtime)
- pytest

## Requirements

- Python 3.10 or newer
- Git
- Internet access for the initial dataset download
- At least 2 GiB of free space where the database will be written, to allow for
  database creation and ingestion headroom

The pinned source archive is approximately 53.4 MB. The generated SQLite
database was approximately 1.26 GB in the verified environment. Actual disk
requirements may vary.

## Run locally (Windows PowerShell)

Run these commands from the repository root.

### 1. Clone the repository

```powershell
git clone https://github.com/salmanfarismkn/Recipe-Intelligence-Assistant.git
Set-Location Recipe-Intelligence-Assistant
```

### 2. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks virtual-environment activation, run this in the current
PowerShell process, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

### 4. Download the dataset

```powershell
python scripts\download_dataset.py
```

This downloads and verifies the pinned dataset archive at
`data\raw\recipes_raw.zip`.

### 5. Ingest the dataset

```powershell
python scripts\ingest_dataset.py data\raw\recipes_raw.zip
```

This initializes the SQLite database, processes the source records, preserves
original recipe information, stores normalized ingredient metadata, and
prepares retrieval indexes. Dataset ingestion is required before starting the
application.

### 6. Start the application

```powershell
uvicorn recipe_intelligence.api:app --host 127.0.0.1 --port 8000
```

Open these addresses in a browser:

| URL | Purpose |
|---|---|
| [http://127.0.0.1:8000/](http://127.0.0.1:8000/) | Web chat |
| [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) | Interactive API documentation |
| [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) | Application and database health |

The default database location is `data\processed\recipes.sqlite3`. To use a
different database location, set `RECIPE_DATABASE_PATH` in the server
environment before starting the application.

## Running tests

With the virtual environment activated:

```powershell
python -m pytest -q
```

The verified test suite contains 72 passing tests.

## Optional: Google Gemini integration

Gemini integration is optional at runtime. Deterministic recipe search,
analytics, matching, and substitution do not require Gemini credentials.

To enable the provider, configure these environment variables in PowerShell:

```powershell
$env:LLM_ENABLED = "true"
$env:GEMINI_API_KEY = "your-api-key"
$env:GEMINI_MODEL = "your-configured-model"
```

Start the application in the same terminal session. Keep API credentials out of
source code and frontend files.

Gemini assists with natural-language interpretation and response framing.
Recipe retrieval, statistical calculations, hard constraints, ranking, and
substitution evidence remain controlled by application logic. Live Gemini
connectivity has not been verified for this repository snapshot.

## Architecture and design documentation

- [Architecture](docs/architecture.md)
- [Design decisions and limitations](docs/design-decisions.md)
- [API and conversation management](docs/api-and-conversations.md)
- [Dataset inventory and ingestion assumptions](docs/dataset-inventory.md)
- [Gemini integration](docs/gemini-integration.md)
- [Substitution evidence model](docs/substitution-engine.md)
- [Evaluation methodology and results](docs/evaluation.md)

Machine-readable reports:

- [Dataset inventory](reports/dataset_inventory.json)
- [Evaluation report](reports/evaluation_report.json)

## Dataset and implementation notes

The application processes three source-specific JSON maps containing 125,164
recipe IDs. Original recipe records and ingredient strings are preserved;
normalized ingredient representations and derived metadata are stored
separately.

Important limitations:

- Search uses deterministic SQLite FTS5 keyword retrieval, not semantic vector
  search.
- Ingredient quantities are not verified for sufficiency during matching.
- Ingredient-text restrictions are not medical or allergy-safety guarantees.
- Substitution recommendations depend on available local recipe evidence and
  may refuse unsupported adaptations.
- Instruction-step counts are derived from the dataset's stored instruction
  structure and have formatting-related limitations.
- Session message ordering is coordinated within a single API process;
  distributed multi-worker coordination is not implemented.

The application does not rely on an LLM to calculate dataset statistics,
enforce hard matching constraints, or establish substitution evidence.
