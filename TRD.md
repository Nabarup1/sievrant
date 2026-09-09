# Sievrant: Technical Requirements Document

**Version**: 1.0
**Date**: August 30, 2026
**Status**: Draft

---

## 1. Math Inventory

Every mathematical component in Sievrant is either borrowed from established literature or is a novel application of borrowed math. There is no invented math. The defensible IP is the combination, the productization, and the Evidence Packet contract.

| Layer | Math | Status | Citation | What it replaces |
|---|---|---|---|---|
| Vector index | Vamana graph, alpha RNG pruning, FreshDiskANN streaming merge | Borrowed (Microsoft Research, open source) | Subramanya et al. 2019 (DiskANN), Singh et al. 2021 (FreshDiskANN) | Infrastructure, no replacement |
| Lexical index | BM25 (Okapi) | Borrowed, decades old, still best in class for exact match | Robertson and Zaragoza 2009 | Infrastructure |
| Cross source fusion | Reciprocal Rank Fusion: score is the sum of 1/(k+rank) where k is 60 | Borrowed, industry default | Cormack, Clarke, Buettcher 2009 | The "which index do I trust" LLM guess |
| Cross encoder reranking | BERT based cross encoder or ColBERT late interaction scoring | Borrowed, production standard | Nogueira and Cho 2019 (cross encoder), Khattab and Zaharia 2020 (ColBERT) | Raw similarity scores from first stage retrieval |
| Evidence selection | Monotone submodular maximization under knapsack constraint, greedy (1 minus 1/e) bound | Borrowed, newly applied to token budget | Nemhauser, Wolsey, Fisher 1978 | Top k guess, redundant evidence |
| Stopping and trust | Conformal prediction: distribution free, finite sample coverage guarantee via calibrated non conformity scores | Borrowed math, novel application as always on product gate | Vovk, Gammerman, Shafer 2005; Angelopoulos and Bates 2021 | The "LLM feels confident" guess |
| Graph reasoning | Personalized PageRank: (I minus alpha times D inverse times A) times x is (1 minus alpha) times v | Borrowed | Page et al. 1999; HippoRAG (Fang et al. 2024) | Single hop tree only reasoning |
| Query routing | TF IDF based complexity classifier with confidence gating | Borrowed, applied to RAG routing | Standard IR, validated by RAGRouter Bench 2026 | Fixed pipeline for all queries |
| Semantic caching | Cosine similarity over query embeddings with threshold gating | Standard technique | GPTCache, industry practice | No caching (full pipeline every time) |

### 1.1 The Conformal Prediction Detail

This is the core differentiator and deserves precise specification.

**What it is**: A distribution free method for constructing prediction sets with finite sample coverage guarantees. Given a calibration set of (query, relevant_chunks) pairs, compute non conformity scores for each chunk. At inference time, include a chunk in the evidence set only if its non conformity score falls below a threshold determined by the calibration quantile.

**The guarantee**: For a user specified coverage level alpha (e.g., 0.95), the probability that the true relevant chunk is included in the returned set is at least 1 minus alpha, under the exchangeability assumption.

**The exchangeability caveat**: This guarantee holds when the calibration distribution matches the test distribution. In production, this means:
- Per tenant calibration sets are operationally necessary
- Live monitoring of empirical coverage vs nominal is non negotiable
- Drift detection triggers recalibration automatically
- The `empirical_coverage_last_30d` field in the Evidence Packet is the honesty mechanism

**Non conformity score computation**: For each candidate chunk c given query q:
```
s(q, c) = 1 - sim(embed(q), embed(c)) * relevance_signal(c)
```
Where `relevance_signal` can incorporate reranker scores, structural position, and freshness. Lower scores mean higher confidence that the chunk is relevant.

---

## 2. System Pipeline: 10 Stage Retrieval

The retrieval pipeline executes in this order for every query:

```
Query In
  |
  v
[1. Cache Check] -- hit --> return cached Evidence Packet
  |
  miss
  v
[2. Query Processing] -- expansion, decomposition, embedding
  |
  v
[3. Adaptive Routing] -- classify complexity --> FAST or SLOW path
  |
  v
[4. ACL Resolution] -- resolve user identity to allowed ACL groups
  |
  v
[5. Multi Index Retrieval] -- vector + BM25 + tree + graph (parallel, ACL filtered)
  |
  v
[6. RRF Fusion] -- merge ranked lists from all indexes
  |
  v
[7. Cross Encoder Reranking] -- score top N candidates with cross encoder
  |
  v
[8. Submodular Selection] -- greedy selection under token budget
  |
  v
[9. Conformal Gate] -- filter by non conformity threshold, decide ANSWERED or ABSTAINED
  |
  v
[10. LLM Generation] -- produce answer with citations from selected evidence
  |
  v
Evidence Packet Out (cached for future queries)
```

### 2.1 Stage Details

**Stage 1: Cache Check**
- Exact match on query hash (sub millisecond)
- Semantic match: cosine similarity of query embedding against cache index (threshold 0.97 default)
- Tenant isolated: cache lookups scoped by tenant ID
- Freshness aware: cached entries expire based on source document volatility

**Stage 2: Query Processing**
- Query expansion via LLM (add synonyms, rephrase for recall)
- HyDE (Hypothetical Document Embeddings): generate a hypothetical answer, embed it, use that embedding for retrieval
- Multi hop decomposition: break compound questions into sub queries
- Embedding generation for vector search

**Stage 3: Adaptive Routing**
- Lightweight classifier (TF IDF features, not embedding based per RAGRouter Bench findings)
- Three routes: FAST (single pass hybrid search), STANDARD (full pipeline), SLOW (multi hop with iterative retrieval)
- Confidence gating: if classifier is uncertain, default to STANDARD
- Cost tracking: route selection recorded in audit block

**Stage 4: ACL Resolution**
- Resolve user identity to set of allowed ACL groups
- Pre retrieval enforcement: ACL groups become metadata filters on all index queries
- No post filtering: unauthorized chunks never enter scoring

**Stage 5: Multi Index Retrieval**
- Vector index (Qdrant/DiskANN): semantic similarity search
- BM25 index (Tantivy): exact match lexical search
- Tree index: hierarchical document structure traversal
- Graph index (PPR): entity relationship traversal
- All queries execute in parallel via tokio
- All queries carry ACL filters as mandatory metadata predicates
- Each returns a ranked list of (chunk_id, score) pairs

**Stage 6: RRF Fusion**
- Merge all ranked lists using Reciprocal Rank Fusion
- Formula: `fused_score(d) = sum over all lists L of 1/(k + rank_L(d))` where k is 60
- Deterministic, no score calibration needed across heterogeneous indexes
- Invariant: rank monotonicity (tested via proptest)

**Stage 7: Cross Encoder Reranking**
- Take top N candidates from RRF (N defaults to 100)
- Score each (query, chunk) pair using cross encoder model
- Options: local cross encoder via ONNX Runtime, or API call to Cohere Rerank / similar
- ColBERT late interaction as lower latency alternative
- Output: reranked list with rerank_scores

**Stage 8: Submodular Selection**
- Greedy algorithm: iteratively select the chunk with highest marginal gain
- Marginal gain function: combines relevance (rerank score), diversity (1 minus max_sim to already selected), and information content
- Constraint: total tokens of selected chunks must not exceed budget
- Guarantee: achieves at least (1 minus 1/e) approximation to optimal
- Invariant: never exceeds token budget (tested via proptest)

**Stage 9: Conformal Gate**
- Compute non conformity score for each selected chunk
- Compare against calibration threshold for the target coverage level
- Decision: ANSWERED if sufficient evidence passes the gate, ABSTAINED otherwise
- On abstention: return the Evidence Packet with `decision: "ABSTAINED"` and the reason

**Stage 10: LLM Generation**
- Construct prompt with selected evidence chunks and original query
- Call LLM via the provider trait (OpenAI compatible, Anthropic native, Gemini native)
- Extract answer and citation spans
- Assemble final Evidence Packet

---

## 3. Dependency Strategy

| Component | Approach | Crate/Library | Version | Reasoning |
|---|---|---|---|---|
| Vector ANN | Depend on | `qdrant-client` | latest | Mature, benchmarked, zero differentiation in rebuilding |
| BM25 Lexical | Depend on | `tantivy` | latest | Rust native, production grade, exact match is solved |
| Graph primitives | Depend on | `petgraph` | latest | Graph traversal is solved; PPR over our schema is not |
| Cross encoder ONNX | Depend on | `ort` (ONNX Runtime) | latest | Local inference for reranking without API calls |
| Async runtime | Depend on | `tokio` | 1.x | Industry standard async runtime |
| HTTP client | Depend on | `reqwest` | latest | Async HTTP is solved infrastructure |
| Serialization | Depend on | `serde`, `serde_json` | latest | Universal Rust serialization |
| Config | Depend on | `config` | latest | Layered config from multiple sources |
| Error handling | Depend on | `thiserror`, `anyhow` | latest | Typed errors for libraries, anyhow for applications |
| CLI | Depend on | `clap` | latest | Argument parsing |
| Tracing/Observability | Depend on | `tracing`, `opentelemetry` | latest | Structured logging and OTel spans |
| Property testing | Depend on | `proptest` | latest | Property based testing for invariants |
| Benchmarking | Depend on | `criterion` | latest | Statistical micro benchmarks |
| UUID generation | Depend on | `uuid` | latest | Query and trace ID generation |
| Timestamp | Depend on | `chrono` | latest | Timestamp formatting |
| Hashing | Depend on | `sha2` | latest | Document version hashing, audit trail |
| RRF fusion | Write it | Custom, approx 40 lines | n/a | Sum of reciprocals, a dependency is overhead |
| Submodular selection | Write it | Custom, greedy loop | n/a | Core logic, keep inspectable |
| Conformal gate | Write it | Custom, minimal stats | n/a | Core differentiator, own and audit every line |
| PPR power iteration | Write it | Custom on top of petgraph | n/a | PPR over our schema is our logic |
| Semantic cache | Write it | Custom with embedding index | n/a | Tight integration with retrieval pipeline |
| ACL enforcement | Write it | Custom metadata filter layer | n/a | Security critical, must own |
| Query router | Write it | Custom TF IDF classifier | n/a | Lightweight, domain specific |
| Audit trail | Write it | Custom tamper evident logging | n/a | Compliance critical |

---

## 4. LLM Provider Abstraction

### 4.1 Trait Design

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse>;
    fn supports_vision(&self) -> bool;
    fn context_window(&self) -> usize;
    fn model_id(&self) -> &str;
}
```

### 4.2 Adapter Catalog

| Provider | Adapter struct | Format | Notes |
|---|---|---|---|
| OpenAI | `OpenAiCompatible` | OpenAI chat completions | Reference format |
| DeepSeek | `OpenAiCompatible` | OpenAI compatible | Change base_url and model only |
| GLM (Zhipu) | `OpenAiCompatible` | OpenAI compatible | Via gateway or direct |
| Anthropic (Claude) | `AnthropicNative` | Anthropic Messages API | Distinct schema, native adapter for full feature access |
| Gemini | `GeminiNative` | Gemini API | Native adapter for grounding/search features |

### 4.3 Request/Response Types

```rust
pub struct LlmRequest {
    pub messages: Vec<Message>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub system_prompt: Option<String>,
}

pub struct LlmResponse {
    pub content: String,
    pub usage: TokenUsage,
    pub model: String,
    pub finish_reason: FinishReason,
}

pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}
```

---

## 5. Performance Budgets

### 5.1 Per Stage Latency Budget (p50, fast path)

| Stage | Budget |
|---|---|
| Cache check | less than 2ms |
| Query processing (embedding only, no expansion) | less than 15ms |
| Routing classifier | less than 5ms |
| ACL resolution | less than 3ms |
| Vector search | less than 20ms |
| BM25 search | less than 10ms |
| RRF fusion | less than 2ms |
| Cross encoder rerank (top 50) | less than 40ms |
| Submodular selection | less than 5ms |
| Conformal gate | less than 2ms |
| Total (fast path, no LLM generation) | less than 100ms |

### 5.2 Memory Budget

| Component | Target |
|---|---|
| Vector index (per 1M docs) | less than 2GB resident (disk backed) |
| BM25 index (per 1M docs) | less than 500MB |
| Graph index (per 1M docs) | less than 1GB |
| Semantic cache (per 100K entries) | less than 200MB |
| Runtime overhead | less than 100MB base |

---

## 6. Streaming Merge and Compounding

### 6.1 LSM Style Tiered Merge

Documents are ingested into an in memory buffer (Level 0). When the buffer fills, it is flushed to disk as a sorted run. When N runs accumulate at a level, they are merged into a single run at the next level. This applies to all four indexes simultaneously.

### 6.2 Invariants

- Per document insert cost: O(log N) amortized
- No query downtime during merge (readers see consistent snapshot)
- Crash safety: WAL records all mutations before they are applied
- Kill mid merge: on restart, detect incomplete merge via WAL, either complete or roll back

### 6.3 Generation Tracking

Each merge cycle increments the `index_generation` counter. The Evidence Packet includes this value so consumers can detect whether their cached results refer to the current or a previous index state.

---

## 7. Calibration Set Management

### 7.1 Requirements

- Minimum calibration set size: 100 labeled (query, relevant_chunks) pairs per tenant
- Recommended: 500+ pairs for stable coverage estimates
- Format: JSONL with query, list of relevant chunk IDs, and optional difficulty tags

### 7.2 Workflow

1. Tenant provides labeled data during onboarding
2. System computes non conformity scores on calibration set
3. Quantile threshold is derived for the target coverage level
4. Threshold is stored per tenant, per coverage level
5. Live queries use this threshold in the conformal gate
6. Empirical coverage is tracked continuously
7. When empirical coverage deviates from nominal by more than the tolerance (default 2 percentage points), an alert fires and recalibration is suggested

### 7.3 Default Calibration

For tenants without labeled data, a default calibration set (constructed from FinanceBench, HotpotQA, and internal benchmarks) provides a reasonable starting threshold. The Evidence Packet clearly marks this as `calibration_set_id: "default"` so consumers know the coverage guarantee is based on generic data, not their specific corpus.

---

## 8. Error Handling Strategy

### 8.1 Error Hierarchy

```rust
// Library errors (for consumers)
#[derive(Debug, thiserror::Error)]
pub enum SievrantError {
    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    #[error("LLM provider error: {0}")]
    Llm(#[from] LlmError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Access denied: user lacks required ACL groups")]
    AccessDenied,

    #[error("Calibration error: {0}")]
    Calibration(#[from] CalibrationError),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),
}
```

### 8.2 Graceful Degradation

- If vector index is unavailable: proceed with BM25 + tree + graph, note degradation in packet
- If cross encoder is unavailable: skip reranking, use RRF scores directly, note in packet
- If LLM is unavailable: return evidence without generated answer, set `decision: "EVIDENCE_ONLY"`
- If cache is unavailable: bypass cache, proceed with full pipeline

---

## 9. Storage Layout (On Disk)

```
corpus_root/
  metadata.json              # corpus metadata, version, config
  wal/
    wal_000001.log           # write ahead log segments
    wal_000002.log
  vector/
    level_0/                 # in memory buffer flush
    level_1/                 # merged runs
    level_2/
  bm25/
    tantivy_index/           # tantivy managed directory
  tree/
    tree_000001.bin          # serialized tree structures
  graph/
    nodes.bin                # petgraph serialized
    edges.bin
    ppr_cache.bin            # precomputed PPR vectors
  cache/
    semantic_index.bin       # query embedding index for cache lookup
    entries/                 # cached Evidence Packets
  calibration/
    default.jsonl            # default calibration set
    tenant_abc.jsonl         # per tenant calibration data
    thresholds.json          # computed quantile thresholds
  audit/
    audit_2026_08.log        # monthly audit log files
    checksums.json           # tamper detection checksums
```

---

## 10. Concurrency Model

### 10.1 Runtime

- tokio multi threaded runtime for all async operations
- Read/write isolation: readers acquire a snapshot, writers mutate through WAL
- No global locks on the hot path (retrieval)
- Write serialization: single writer thread for WAL append, concurrent readers

### 10.2 Index Access Pattern

- All four indexes support concurrent reads
- Writes are batched and flushed periodically
- Merge operations run on a background tokio task
- Merge does not block reads (new readers see old snapshot until merge completes and atomically swaps)
