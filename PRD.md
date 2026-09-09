# Sievrant: Product Requirements Document

**Version**: 1.0
**Date**: August 30, 2026
**Status**: Draft

---

## 1. Product Vision

Sievrant is a **statistically audited retrieval decision engine** built in Rust. It occupies the layer between vector database infrastructure (Qdrant, Pinecone, Milvus) and application orchestration frameworks (LlamaIndex, LangChain, Rig). No product currently exists in this layer.

The core promise: **AI agents get context with full metadata, accurately and fast, with mathematically bounded guarantees on what they receive.**

Sievrant is not another vector database. It is not another agent framework. It is the calibrated decision layer that currently exists only as scattered academic papers and that no one has shipped as a product.

### 1.1 One Sentence

Sievrant is a Rust library that retrieves evidence from document corpora with provable coverage guarantees, formal trust scoring, and a structured Evidence Packet that agents can audit and act on programmatically.

### 1.2 The Problem

Every production RAG system in 2026 suffers from the same set of decisions that are currently delegated to LLM guesswork:

| Decision point | Current approach | Why it fails |
|---|---|---|
| Which index to trust (vector vs BM25 vs tree vs graph) | Agent "reasons" about which tool to call | Non deterministic, varies run to run, no calibration |
| Which evidence chunks make the final context | Top k by similarity score | Redundant chunks waste tokens, scores not comparable across indexes |
| When to stop retrieving and answer | LLM "feels" confident | No statistical basis, no audit trail, fails silently |
| Whether to trust a retrieved chunk at all | LLM eyeballs it | No formal bound on error rate |

These are not LLM problems. They are decision theory problems with known, provable solutions that nobody has productized.

### 1.3 The Solution

Replace every LLM guess with a closed form or provably bounded operation:

| Decision point | Sievrant mechanism | Mathematical basis |
|---|---|---|
| Which index to trust | Reciprocal Rank Fusion (RRF) | Rank based, needs no score calibration, deterministic (Cormack et al. 2009) |
| Which chunks make the final context | Submodular greedy selection under token budget | Provable (1 minus 1/e) approximation to optimal set (Nemhauser et al. 1978) |
| When to stop retrieving | Conformal prediction coverage gate | Distribution free, finite sample statistical guarantee |
| Whether to trust a chunk | Conformalized retrieval scoring | Formally bounded probability of being off source |

The LLM's role shrinks to exactly two places: understanding the query (expansion, decomposition) and writing the final answer with citations. It never decides routing, selection, or stopping.

---

## 2. Target Users

### 2.1 Primary: AI Agent Developers

Engineers building compound AI systems who need a retrieval backend that:
- Returns structured, machine readable evidence (not text blobs)
- Provides auditable confidence scores with mathematical backing
- Works as a library (embedded) or service (HTTP/gRPC/MCP)
- Does not require them to understand RRF, submodular optimization, or conformal prediction to get good defaults

### 2.2 Secondary: Enterprise RAG Teams

Teams in regulated industries (legal, healthcare, finance) who need:
- Permission aware retrieval that enforces ACLs before scoring, not after
- EU AI Act compliant audit trails (Articles 12/13)
- Immutable provenance for every retrieval decision
- Per tenant calibration with live coverage monitoring

### 2.3 Tertiary: RAG Infrastructure Teams

Platform engineers who currently assemble retrieval from Qdrant plus LangChain plus custom reranking plus ad hoc evaluation and want a single, coherent, benchmarked alternative.

---

## 3. Core User Stories

### 3.1 The One Liner

**As an agent developer**, I want to call `sievrant::retrieve("What is the lease term?", &corpus).await?` and receive a structured Evidence Packet with coverage guarantees, so that my agent can make grounded decisions without understanding the retrieval internals.

### 3.2 The Tunable Call

**As a power user**, I want to adjust coverage target, token budget, LLM provider, and routing strategy through a builder API, so that I can optimize for my specific latency/accuracy tradeoff.

```rust
let packet = sievrant::retrieve("What is the lease term?", &corpus)
    .coverage(0.95)
    .budget(4000)
    .llm(Provider::Claude)
    .route(Route::Auto)
    .await?;
```

### 3.3 Permission Aware Retrieval

**As an enterprise admin**, I want retrieval to enforce document level ACLs before any scoring happens, so that unauthorized users never see chunks they should not access, even in vector similarity scores.

### 3.4 Compliance Audit

**As a compliance officer**, I want every retrieval to produce an immutable audit record containing the model version, retrieval stages with timing, calibration set ID, and a tamper evident session hash, so that I can satisfy EU AI Act Article 12 requirements.

### 3.5 Agent Integration via MCP

**As an agent framework developer**, I want to expose Sievrant as an MCP server so that any MCP compatible agent (Claude, Cursor, VS Code agents, custom) can use it as a retrieval tool without custom integration code.

### 3.6 Compounding Corpus

**As a data engineer**, I want to continuously add documents to a corpus without full reindexing, with per document insert cost that stays O(log N) amortized at 100M documents, so that the system handles growing knowledge bases without degradation.

### 3.7 Live Coverage Monitoring

**As an ML engineer**, I want to see the empirical coverage rate over the last 30 days alongside the nominal coverage target, and get alerts when they diverge beyond tolerance, so that calibration drift is caught before it affects users.

---

## 4. The Evidence Packet: Product Surface Contract

The Evidence Packet is the actual product. Everything upstream exists to fill it correctly. Every field exists because a specific consumer needs it.

```json
{
  "query_id": "q_8f2a...",
  "trace_id": "otel_abc123...",
  "answer": "The lease term is 60 months, per Section 4.2.",
  "decision": "ANSWERED",
  "coverage_guarantee": 0.95,
  "empirical_coverage_last_30d": 0.947,
  "route": "FAST",
  "cache_hit": false,
  "evidence": [
    {
      "chunk_id": "doc_449:sec_4.2:p3",
      "source_doc": "MasterLease_2026.pdf",
      "doc_version_hash": "sha256:9f2e...",
      "path": ["Article 4", "4.2 Term", "paragraph 3"],
      "page": 14,
      "char_span": [1882, 2043],
      "retriever": "tree_index",
      "rrf_score": 0.0164,
      "rerank_score": 0.92,
      "nonconformity_score": 0.03,
      "submodular_marginal_gain": 0.41,
      "ingested_at": "2026-08-14T09:02:11Z",
      "index_generation": 1188,
      "acl_groups": ["legal", "executives"],
      "freshness_hours": 12.4
    }
  ],
  "audit": {
    "model_version": "claude-sonnet-4-20250514",
    "retrieval_stages_ms": {
      "cache_check": 2,
      "routing": 5,
      "vector_search": 45,
      "bm25_search": 12,
      "rrf_fusion": 3,
      "reranking": 67,
      "submodular_selection": 8,
      "conformal_gate": 4,
      "llm_generation": 68
    },
    "total_candidates_before_rerank": 120,
    "total_candidates_after_rerank": 25,
    "total_candidates_after_conformal": 8,
    "calibration_set_id": "cal_set_v3_legal",
    "session_hash": "sha256:audit_trail_hash"
  },
  "latency_ms": 214,
  "cost_usd": 0.0021
}
```

### Field Justification

| Field | Consumer | Why it exists |
|---|---|---|
| `doc_version_hash` | Agent | Detect if source changed since retrieval (critical for compounding corpora) |
| `path` + `char_span` | Citation UI | PageIndex style micro citation traceability |
| `retriever` + scores | Downstream agent | Apply its own trust policy instead of taking your word for it |
| `coverage_guarantee` vs `empirical_coverage_last_30d` | ML engineer | Honesty check: is the guarantee actually holding |
| `acl_groups` | Security | Verify authorization was enforced |
| `trace_id` | Observability | Correlate with OpenTelemetry spans |
| `route` | Cost tracking | Know if query took fast or slow path |
| `audit` block | Compliance | EU AI Act Articles 12/13 |

---

## 5. Non Functional Requirements

### 5.1 Latency Targets

| Path | p50 | p95 | p99 |
|---|---|---|---|
| Cache hit | less than 5ms | less than 10ms | less than 20ms |
| Fast path (simple query) | less than 100ms | less than 200ms | less than 500ms |
| Slow path (multi hop) | less than 500ms | less than 1500ms | less than 3000ms |

### 5.2 Throughput

- Sustain 1000+ queries per second on a single node for cache hits
- Sustain 100+ queries per second for fast path on a single node
- Sustain 20+ queries per second for slow path (LLM bound)

### 5.3 Scale

- Single corpus up to 100M documents
- Per document insert cost: O(log N) amortized
- Memory usage: sub linear in corpus size (disk backed indexes)

### 5.4 Correctness

- Conformal coverage: empirical coverage within 2 percentage points of nominal target over any 30 day window
- RRF invariant: rank monotonicity (a doc cannot rank worse in fused list than in every individual list)
- Submodular invariant: selected evidence never exceeds token budget
- Zero unauthorized chunk leakage (pre retrieval ACL enforcement)

### 5.5 Reliability

- Crash recovery: zero data loss on process kill during merge
- WAL for all index mutations
- Graceful degradation: if one index is unavailable, others still serve (with coverage note in packet)

---

## 6. Competitive Landscape

| Player | Category | What they are | Gap vs Sievrant |
|---|---|---|---|
| Qdrant | Infrastructure | Rust vector DB, $87.8M funding | No decision layer, no reranking, no conformal gate, no compliance |
| Pinecone | Infrastructure | Managed vector DB, ~$750M valuation | Same gaps plus vendor lock in at scale |
| Weaviate | Infrastructure | Hybrid search vector DB | Infrastructure only |
| Milvus/Zilliz | Infrastructure | Open source vector DB, $113M funding | Infrastructure only |
| Vectara | Managed RaaS | End to end managed RAG | Opaque, no statistical guarantees, no self hosted, no Evidence Packet |
| Glean | Turnkey Enterprise | Enterprise search, $50 to $75/user/month | Closed, not embeddable, not for agent developers |
| Cohere | Model Layer | Embed, Rerank, Command models | Model provider, not a decision layer |
| LlamaIndex | Orchestration | Python framework | Glue code, no math, no guarantees |
| LangChain | Orchestration | Python framework | Same |
| RAGFlow | Open Source RAG | Deep document understanding | Python, no formal guarantees |
| Rig | Rust AI | Rust LLM agent framework | Orchestration only, no retrieval engine |
| RustyRAG | Rust RAG | Rust RAG API, sub 200ms | Missing conformal gate, graph, submodular, compliance |
| PageIndex | Tree Index | Tree plus LLM navigation | No graph, no formal stopping, single hop |
| Microsoft GraphRAG | Graph RAG | Graph based retrieval | Indexing costs hit five figures, not for compounding corpora |

### Positioning Statement

Sievrant is the calibrated decision layer that sits between vector databases and agent applications. It is the first product to ship conformal prediction as a first class, always on trust gate in commercial retrieval. The Evidence Packet is not just an API response format; it is a compliance artifact.

---

## 7. Pricing Model Considerations

| Model | Target | Structure |
|---|---|---|
| Open Core | Developers, startups | Free self hosted library. Core retrieval plus conformal gate plus RRF plus submodular selection. |
| Pro (self hosted) | Growth teams | Paid features: semantic cache, MCP server, advanced observability, multi tenant ACLs |
| Enterprise | Regulated industries | Per corpus licensing. Compliance trail, calibration onboarding, SLA on coverage guarantee monitoring, priority support. |
| Cloud (future) | Teams wanting managed | Usage based: per query plus per GB indexed. Similar to Vectara model but with statistical guarantees as the differentiator. |

---

## 8. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Exchangeability caveat: conformal guarantee relies on calibration set matching production traffic | High | Live coverage monitoring (empirical_coverage_last_30d), automated recalibration triggers, explicit drift alerts |
| Per tenant calibration is ongoing operational burden | Medium | Onboarding flow with minimum data requirements, automated re calibration on schedule, default calibration set as fallback |
| Thin technical moat: RRF, submodular, conformal are published math | High | Execution speed, Evidence Packet becoming de facto standard, compliance integration, being first to make "statistically audited retrieval" a category |
| One line SDK defaults might not work across varied corpora | Medium | Default tuning research across benchmark suites before launch, progressive disclosure (sane defaults but tunable) |
| Rust slows initial velocity vs Python research loop | Medium | Lean on mature deps (Qdrant, Tantivy, petgraph), write only differentiated code in Rust, Python tooling for benchmarks and evaluation |
| Conformal coverage claim quietly stops being true if traffic drifts | High | The empirical_coverage_last_30d field in every packet, dashboard monitoring, automated alerts, documented as a product guarantee, not just a number |

---

## 9. Success Metrics

| Metric | Target (6 months post launch) | Measurement |
|---|---|---|
| GitHub stars | 2000+ | Public repo |
| Monthly active library users | 500+ | Telemetry (opt in) |
| Enterprise pilot customers | 3 to 5 | Sales pipeline |
| FinanceBench accuracy | greater than 90% (with confidence intervals) | Automated benchmark CI |
| p50 latency fast path | less than 100ms | Criterion benchmarks |
| Conformal coverage empirical accuracy | Within 2 percentage points of nominal | Live monitoring |
| Evidence Packet adoption | 2+ agent frameworks consuming natively | Partnership / integration |

---

## 10. Research Track (Explicitly Not in Core Product)

These are real and valuable but require LLM as simulator running thousands of rollouts, which is not shippable as a product today:

- **POMCP / PO UCT planning**: Deep planning for query strategy optimization
- **Particle filter belief tracking**: Probabilistic state estimation during multi hop retrieval
- **Query conditioned dynamic PPR reweighting**: Adapting graph traversal weights per query at inference time

They belong in a research crate that can be opted into experimentally but are not part of the core retrieve() pipeline.
