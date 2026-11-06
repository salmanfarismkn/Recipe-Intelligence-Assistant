# Evaluation and reproducibility

## Snapshot

The evaluation uses the pinned Eight Portions archive:

- SHA-256: `1ae20f3260313d501edf23c22c6f1875e87917f5a8e2ff13b40450719322c81b`
- Records: 125,164 total; AR 39,802, Epi 25,323, FN 60,039
- Stable IDs: 125,164 unique across the three source partitions

See [dataset inventory](./dataset-inventory.md) for schema and data-quality
counts. The JSON snapshot is preserved in
[`reports/dataset_inventory.json`](../reports/dataset_inventory.json).

## Reproduce the deterministic evaluation

Install the test extras, download and ingest the pinned data, run tests, and
generate a full-dataset report:

```powershell
python -m pip install -e ".[test]"
python scripts\download_dataset.py
python scripts\ingest_dataset.py data\raw\recipes_raw.zip
python -m pytest -q
python scripts\evaluate_llm_grounding.py data\raw\recipes_raw.zip
python scripts\run_release_evaluation.py --database data\processed\recipes.sqlite3 --repeats 3 --output reports\evaluation_report.json
```

The first evaluator ingests the archive into a temporary in-memory database
and checks typed-entity validation, verified result references, fabricated
fact-ID rejection, and hard-exclusion preservation. The release runner opens
the ingested SQLite database read-only, checks source counts and stored-value
integrity, then warms each service and records three local timing samples.
Performance medians are hardware/cache dependent, not service-level objectives.

The generated [release report](../reports/evaluation_report.json) records the
exact host/runtime, database size, integrity checks, candidate counts, and
service timings from the run. The downloaded archive and database are ignored
local data, not checked-in fixtures.

## Verified behavior

- `python -m pytest -q`: **72 passed** in 4.30 seconds; two Starlette/httpx
  deprecation warnings.
- Full archive grounding evaluator: all four checks passed on all 125,164
  records. It verified three actual search references, analyzed 37,344
  matching candidates, preserved a peanut exclusion, and rejected an invented
  fact ID.
- A fresh Uvicorn process became ready in 1.133 seconds on the recorded host.
  `/health`, `/`, `/assets/app.js`, and a live `penne` search returned `200`.
- Full-dataset HTTP API checks returned `200` for search, garlic-prevalence
  analytics, matching, and substitution. Prevalence was 37,344 / 124,647 valid
  ingredient-list recipes; matching examined 72,542 candidates and returned
  five; the substitution service returned ten unsupported evidence options
  rather than asserting an unevidenced adaptation.
- Full-dataset conversation exercised: request matching with eggs, onions,
  tomatoes, rice, peanut exclusion and a six-step ceiling; add the
  fewer-additional preference; exclude eggs; show the second recipe; retrieve
  its original instructions. All five turns completed. State retained both
  hard exclusions, the ceiling, the preference and actual result IDs; the
  selected recipe instructions matched the source database and state remained
  readable after the message sequence.
- A regression test verifies `milk-free` rejects oat milk while `dairy-free`
  continues to allow recognized plant milks.
- The pinned archive ingested in 114.354 seconds on this host. The generated
  SQLite database was approximately 1.262 GB with 1,316,950 ingredient rows.

## Performance interpretation

The recorded host was Windows 11, Python 3.12.4, AMD Ryzen 7 5800HS (8 cores,
16 logical CPUs), 16,541,700,096 bytes of physical memory. The report's
benchmarks use one local SQLite file, one warm-up, and three repeated
in-process service calls:

| Operation | Median | Sample output |
|---|---:|---|
| FTS search for `penne` | 0.0101 s | 10 records |
| Garlic prevalence | 8.2713 s | 124,647 valid-list denominator |
| Matching eggs/onions/tomatoes/rice | 6.7901 s | 72,542 candidates, 5 results |
| Cream substitution discovery | 4.7379 s | 10 options; `unsupported` |

These are machine/cache-dependent observations, not latency SLOs. They do not
measure concurrent traffic or an HTTP server under load.

## Verification matrix

| Area | Status | Evidence / limitation |
|---|---|---|
| Dataset counts, IDs, original fields, and normalized rows | PASS | Pinned snapshot inventory and read-only full-database audit; zero source-field or normalization mismatches. |
| Deterministic service and API test suite | PASS | 72 passed; 2 dependency deprecation warnings. |
| Hard exclusions and step ceiling in matching | PASS | Integration/regression tests and full-dataset conversation. |
| Preference effects on ranking without eligibility changes | PASS | Dedicated regression test. |
| Contextual result selection and original instructions | PASS | Full-dataset API conversation; selected ID and instructions resolve to actual prior results. |
| Substitution classifications and restriction failures | PASS with conservative limits | Local evidence and focused tests; no medical/allergy safety claim. |
| Gemini without configured credentials | PASS for deterministic fallback | Live provider verification is not performed. |
| Scoped security review | MANUAL REVIEW ONLY | Bounded inputs, parameterized queries, text-only rendering and no frontend credentials were inspected; no penetration test. |
| Live Gemini model/structured response | NOT VERIFIED | No key/model were configured; provider tests are mocked. |
| Clean install on another machine | NOT VERIFIED | Dependencies and current Python runtime were used locally. |
| Docker/container execution | NOT AVAILABLE | No Dockerfile or Compose configuration exists. |
| Production concurrency, multi-worker deployment, session expiry | NOT VERIFIED / NOT IMPLEMENTED | Locks are process-local; there is no expiry or distributed coordinator. |
| Browser automation/visual inspection | NOT VERIFIED | Fresh-process HTTP/static-asset checks passed; no browser automation was run. |

## Readiness assessment

**Conditionally ready for local deterministic review; not verified as a
production or Gemini-enabled release.** Dataset integrity, deterministic
services, API routes, and the required multi-turn restrictions were exercised
against the full pinned dataset. Remaining verification gaps are live Gemini
access/model/structured output, browser automation, clean install on a separate
machine, and container deployment. Deployment is intentionally local and
single-process until those environments are supplied and evaluated.

## Known limitations

- Keyword search is not semantic retrieval.
- Ingredient parser aliases and quantity handling are intentionally limited.
- Instruction-step counts are syntactic and often ambiguous; unknown counts are
  not eligible under a strict maximum.
- Statistics do not include median or arbitrary ingredient combination
  analyses.
- Substitution comparisons are observational, not controlled substitution
  trials; only strong local evidence can support an adaptation.
- The offline natural-language parser is bounded. Gemini interpretation is
  optional and live behavior has not been measured.
- The browser keeps transcript history locally; the backend has no session-list
  endpoint.
- There is no Docker or production deployment setup.
- There is no authentication. Run only as a local/trusted-user application; do
  not expose the API to an untrusted network without adding and validating an
  appropriate access-control boundary.
