# Sievrant deployment review (2026-11-06)

## Verdict

**Deployable now as a single-instance, API-key-protected REST service**, behind
a TLS-terminating reverse proxy. The statistical core is sound. The HTTP server
was the weak point: it did not compile against the refactored library, and the
previous version was not safe to expose. It has been rewritten and tested.

Not yet ready for: horizontal scaling (multiple replicas), per-key rate limits,
or claims beyond what the guarantee actually covers (see "Statistics" below).

---

## What was wrong with the server, and what changed

| Problem (before) | Now |
|---|---|
| Did not compile (4 errors) against the refactored `sievrant` library | Compiles with zero warnings; 12 server tests + 73 library tests pass |
| Only API key hard-coded to `sk-test` in `main.rs` | Keys from `SIEVRANT_API_KEYS` / `SIEVRANT_API_KEY` / `SIEVRANT_API_KEY_HASHES`; server **refuses to start** with no keys, keys under 16 chars, or example-file placeholders |
| Corpora keyed by name only: any tenant could query any tenant's corpus | Registry keyed by `(tenant, corpus)` on top of the library's tenant-scoped storage; test proves isolation |
| New `Services` (LLM/embedder/reranker clients) built per corpus | Built once at startup and shared |
| Two concurrent requests could open the same corpus twice (two ingest locks over one file set) | Opening is serialised; one instance per corpus |
| `/v1/query` ignored `acl_groups` | ACL groups passed to the pipeline (applied before scoring) |
| `decision` serialised with `Debug` ("Answered"), which did not match the Python SDK enum (`"ANSWERED"`) | Uses the wire format: `ANSWERED` / `ABSTAINED` / `EVIDENCE_ONLY` |
| `coverage_guarantee` always a number, even with no calibration | `null` when no guarantee applies, plus `guarantee_status` and `guarantee_statement` |
| Queries bypassed the audit log (`orchestrator.retrieve` called directly) | Goes through `Corpus::retrieve`, which writes the audit record |
| Ingest `panic!`ed if corpus creation failed; ingested one doc at a time; dropped `acl_groups` and most metadata | Batched ingest, JSON errors, metadata mapped (title, author, created_at, custom keys), ACL groups kept |
| gRPC `Ingest` had no auth, ingested nothing, and reported success | Authenticated, really ingests |
| gRPC `QueryStream` streamed a hard-coded sentence | Returns `UNIMPLEMENTED` |
| Startup printed the full config to stdout (`DEBUG CONFIG`) | Removed; logs a provider summary with no secrets |
| Broken config silently replaced by defaults (`unwrap_or_default`) | Startup fails with the config error |
| Host, port, CORS, body limit, concurrency limit, timeouts in `ServerConfig` all ignored | All honoured. CORS is off unless `SIEVRANT_CORS_ORIGINS` is set (was `permissive`) |
| No graceful shutdown | SIGTERM/Ctrl-C drains in-flight requests |
| Plain-text errors for bad JSON | Every error is `{"error": code, "message": text}` with the right status |
| docker-compose set `SIEVRANT_QDRANT_URL`, which nothing reads, and no backend flag, so "Qdrant" mode actually ran in-memory | `SIEVRANT_VECTOR_BACKEND=qdrant`, `QDRANT_URL=http://qdrant:6334`, persistent data volume, `.env` loaded |
| Dockerfile compiled every workspace crate (incl. Python/Node bindings), ran as root | Builds only `sievrant-server`, runs as a non-root user, has HEALTHCHECK and `/app/data` volume |
| `.env.example` used variable names the config does not read | Rewritten with the real names |

### Endpoints

Public: `GET /v1/health`, `GET /swagger-ui`, `GET /api-docs/openapi.json`

Bearer-key protected:

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/query` | Retrieve (+ answer); summary with guarantee status |
| POST | `/v1/query/full` | Full Evidence Packet (evidence, audit, guarantee, receipt) |
| POST | `/v1/search` | Ranked candidates, no gate or answer |
| POST | `/v1/ingest` | Ingest documents (creates corpus) |
| DELETE | `/v1/corpora/{corpus}/documents/{doc_id}` | Delete a document |
| GET | `/v1/corpora`, `/v1/corpora/{corpus}` | List / stats |
| GET | `/v1/corpora/{corpus}/documents` | Page through documents |
| GET | `/v1/corpora/{corpus}/calibration?coverage=` | Calibration state and thresholds |
| GET | `/v1/corpora/{corpus}/gaps` | Most frequent abstained queries |
| POST | `/v1/calibration` | Calibrate from labelled examples |
| POST | `/v1/calibration/bootstrap` | Synthetic calibration (needs an LLM) |
| POST | `/v1/calibration/feedback` | Recalibrate from collected feedback |
| POST | `/v1/feedback` | Report relevant docs for a served `query_id` |
| GET | `/v1/info` | Active providers (no secrets) and calling tenant |
| POST | `/v1/graph/ingest`, `/v1/tree/summary` | Experimental indexes (409 unless enabled) |

Clients cannot force a route (`FAST`/`SLOW`) or change `rrf_k` / `rerank_top_n`
per request. This is deliberate: those change the score distribution the
calibration was computed on, which would silently void the guarantee.

### Verified

- `cargo test -p sievrant-server`: 12 passed. Includes an HTTP end-to-end test:
  ingest 25 docs, tenant B gets 404 on tenant A's corpus, an uncalibrated query
  returns `coverage_guarantee: null`, calibrate, then a calibrated 90% guarantee,
  feedback accepted, delete idempotency.
- `cargo test -p sievrant -p sievrant-math -p sievrant-calibration -p sievrant-orchestrator`: 73 passed
  (including Monte Carlo coverage and property tests).
- Ran the real binary: refuses to start without keys or with a placeholder key;
  health / 401 / ingest / query / corpora work over curl; SIGTERM exits 0 cleanly.
- **Not run locally:** `docker compose build` (skipped to save disk space).
  First thing to try on a machine with room, or in CI.

### Deploy

```bash
cp .env.example .env
# set SIEVRANT_API_KEYS=yourtenant:$(openssl rand -hex 32), plus provider keys
docker compose up -d --build
curl -s localhost:8080/v1/health
```

Production checklist:
1. Run **exactly one replica** per data volume. Chunk store, BM25 index and
   calibration files are local files with an in-process lock; two replicas on
   the same volume will corrupt them.
2. Terminate TLS in a reverse proxy (Caddy, nginx, a cloud load balancer). The server speaks plain HTTP.
3. Set real embedding and reranker providers. The offline `local` embedder is for tests only (it logs a warning).
4. Calibrate every corpus before advertising a guarantee: at least 100
   labelled queries, ideally a few hundred (see below).
5. Back up the data volume (`/app/data`): it holds documents, indexes and calibration sets.

---

## Statistics: what holds, and what would make it better

### What is correct

`sievrant-math/src/conformal.rs` is textbook split conformal prediction:

- Query score = nonconformity of the best relevant candidate, `+inf` if
  retrieval missed. Threshold = the `ceil((n+1)(1-α))`-th smallest score.
- If the retriever misses too often to reach the target, it reports
  `Unachievable { max_coverage }` instead of returning a threshold that would
  fail silently.
- It refuses to calibrate with too few examples.
- Staleness is detected when the pipeline fingerprint changes or the corpus grows past 1.5×.
- Feedback logging stores all reranked candidates, not just gated ones, so feedback scores are not censored.
- The docs state clearly what is **not** guaranteed (answer correctness, abstention on unanswerable queries).

The guarantee is: *for a new answerable query drawn like the calibration
queries, P(returned conformal set contains ≥1 relevant passage) ≥ target.*
That is real and valuable. Phrases in the README such as "mathematically
correct", "95% CI" and "formally bounded P(off-source)" claim more than that and
should be toned down.

### Improvements, ranked by value for effort

**1. Calibration-conditional (PAC) thresholds. Small change, biggest honesty gain.**

The split-conformal guarantee averages over draws of the calibration set. A
deployment calibrates *once*, and the coverage it actually gets is random,
distributed `Beta(k, n+1-k)`. Exact numbers for a 95% target:

| n (calibration queries) | mean coverage | 5th percentile | P(actual < 93%) | P(actual < 90%) |
|---:|---:|---:|---:|---:|
| 20 (current hard minimum) | 0.952 | 0.861 | 23% | 12% |
| 50 (current recommended) | 0.961 | 0.909 | 13% | 3.4% |
| 100 | 0.950 | 0.911 | 16% | 2.4% |
| 200 | 0.950 | 0.923 | 10% | 0.4% |
| 500 | 0.950 | 0.933 | 2.8% | ~0 |
| 1000 | 0.950 | 0.938 | 0.4% | ~0 |

With 20 examples, one deployment in eight delivers under 90% while saying "95%".

Fix: offer a PAC threshold, "≥95% coverage with probability ≥90% over
calibration". Choose the smallest rank `k` with `P(Binomial(n, 0.95) ≥ k) ≤ 0.10`
instead of `ceil((n+1)·0.95)`. The cost is tiny (n=100: rank 99 instead of 96;
n=1000: 960 instead of 951). It needs n ≥ 45 to exist at all. That is about 20
lines in `conformal.rs` plus a `guarantee_type: marginal | pac` field. It also
answers the open question in `memory.md` about minimum calibration size:
**raise `MIN_ANSWERABLE` from 20 to at least 50, and recommend 200+.**

**2. Make the guarantee cover what is actually returned.**
The conformal set is computed *before* token-budget selection. When the budget
truncates it, the returned evidence is a subset and the guarantee does not
strictly apply. The code flags this honestly (`budget_truncated`). Better:
calibrate the full pipeline, i.e. score whether the *final selected* evidence
contains a relevant passage, and pick the threshold with Learn-then-Test
(Angelopoulos et al., 2021). LTT does not need the selection step to be monotone.

**3. Bound wrong answers on out-of-scope queries.**
Abstention on unanswerable queries is measured, not controlled. Calibration
already collects unanswerable examples. Add a second calibrated rule (for
example on the top candidate's score) and use Learn-then-Test to control *both*
the miss rate on answerable queries and the answer rate on unanswerable ones.
This is the property customers usually mean by "it says I don't know".

**4. Drift alerts that stay valid when checked continuously.**
`observed()` recomputes a fixed-sample 95% Wilson interval every time it is
read. Checking it repeatedly inflates the false-alarm rate well past 5%
(optional stopping). Replace it with an anytime-valid confidence sequence, e.g.
a betting or empirical-Bernstein confidence sequence (Howard et al., 2021;
Waudby-Smith & Ramdas, 2023). Same inputs, same output shape, valid at every
check.

**5. Unbiased feedback labels.**
Recalibrating from feedback uses whichever queries users chose to label, which
breaks exchangeability (the code warns about this). Have the API mark a uniform
random 1–5% of queries as "please label", and recalibrate and monitor only from
those.

**6. Coverage per segment (Mondrian conformal).**
Coverage is marginal across FAST/STANDARD/SLOW routes and query types. One
route can undercover while the average looks fine. Calibrating per route gives
a per-route guarantee at the cost of needing more labels per group.

---

## Remaining gaps (not done in this pass)

- **Rate limiting:** `docs/api-reference/rest-api.md` promises per-key limits
  and `X-RateLimit-*` headers. Only a global concurrency limit (503 + `retry-after`) exists.
- **OpenTelemetry export** is a stub in `sievrant-observability`. Logs only.
- **Semantic cache** is intentionally disabled (`NoopCache`), but the docs and README still advertise it.
- **MCP server** (`sievrant-server/src/mcp.rs`) is a mock and is not served.
- **gRPC** has Retrieve and Ingest only; it is off in `.env.example` (`SIEVRANT_GRPC_ENABLED=false`).
- `GET /v1/corpora` opens every corpus to read stats. Fine for tens of corpora, slow for thousands.
- `docs/api-reference/rest-api.md` and the SDKs do not yet document the new
  fields (`guarantee_status`, `guarantee_statement`) or the calibration/feedback endpoints.
- `crates/sievrant-server/tests.rs` is a broken leftover file (not compiled). Delete it.
- The whole refactor (35+ files) is **uncommitted**. Commit it before deploying.

## Disk usage notes (WSL)

Build output is in `target/` (about 3 GB after this work; `cargo clean` removes
it). Keep builds small:

- Build only what you deploy: `cargo build -p sievrant-server`. Avoid
  `--workspace --all-targets --all-features`, which also builds the PyO3/NAPI
  bindings and criterion benchmarks.
- Add `--offline` when the dependency cache already has everything, so nothing is downloaded.
- Optionally add `[profile.dev] debug = "line-tables-only"` to `Cargo.toml`.
  This cuts debug build size substantially.

WSL stores Linux files in a virtual disk on `C:` that grows but does not
shrink on its own. Space freed inside Linux (for example after `cargo clean`)
does not come back to Windows until the disk is compacted. From an
administrator PowerShell, after `wsl --shutdown`: on recent WSL versions run
`wsl --manage <distro> --set-sparse true`; otherwise use `diskpart`
(`select vdisk file="...\ext4.vhdx"`, then `compact vdisk`).
