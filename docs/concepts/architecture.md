# Architecture

This document describes the full internal architecture of Sievrant's retrieval pipeline. Understanding this document will help you tune performance, interpret results, and extend the system.

---

## Overview

Sievrant processes every query through a **10-stage deterministic pipeline**. Each stage is mathematically justified, has bounded latency, and produces a structured output consumed by the next stage. Nothing is delegated to LLM intuition until the final answer generation step.

```
Query Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Semantic Cache Check                                              │
│  Check 3-tier cache (exact, near-duplicate, semantic) for prior answers.   │
│  HIT → return cached EvidencePacket, skip stages 2-10.                    │
│  MISS → continue to Stage 2.                                               │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Adaptive Query Router                                             │
│  Classify query complexity using a TF-IDF + feature classifier.           │
│  Routes to: Fast (single-pass), Standard (expanded), Slow (multi-hop).    │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 3: Query Processing                                                  │
│  Depending on route:                                                        │
│    Fast: embed query as-is.                                                │
│    Standard: expand query variants + HyDE (Hypothetical Document Embeds). │
│    Slow: decompose into sub-queries via LLM, run multi-hop loop.          │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 4: ACL Resolution                                                    │
│  Resolve the requesting user's identity to permitted ACL groups.           │
│  All subsequent index searches are filtered to only return chunks the      │
│  user is allowed to see. Never returns a chunk the user cannot access.    │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 5: 4-way Parallel Index Search                                       │
│  All four indexes are searched concurrently via tokio::task::JoinSet.     │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Vector Index │  │  BM25 Index  │  │  Tree Index  │  │ Graph Index  │  │
│  │   (Qdrant)   │  │  (Tantivy)   │  │ (Hierarchic) │  │    (PPR)     │  │
│  │              │  │              │  │              │  │              │  │
│  │ Dense neural │  │ Sparse exact │  │ Structural   │  │ Multi-hop    │  │
│  │ similarity   │  │ term overlap │  │ proximity    │  │ relational   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                          │  RankedCandidate[]                               │
└──────────────────────────┼──────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 6: Reciprocal Rank Fusion (RRF)                                      │
│  Merges all four ranked lists into a single unified ranking using:         │
│    score(d) = Σ 1 / (k + rank_i(d))   for each index i                   │
│  where k=60 (Cormack et al. 2009 default).                                │
│  Output: FusedCandidate[] sorted by fused_score descending.               │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 7: Cross-Encoder Reranking                                           │
│  Top-N (default 100) FusedCandidates are passed to a cross-encoder model. │
│  The cross-encoder scores (query, document) pairs jointly, not separately. │
│  This is dramatically more accurate than bi-encoder similarity scores.    │
│  Supports: ONNX local model, Cohere Rerank API, Jina Rerank API.         │
│  Output: RerankedCandidate[] sorted by rerank_score descending.           │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 8: Submodular Evidence Selection                                     │
│  Greedy selection under token budget constraint using a facility location  │
│  function that maximizes coverage while minimizing redundancy.             │
│  Algorithm: Lazy Greedy with priority queue — O(N log N).                 │
│  Guarantee: (1 - 1/e) ≈ 63% of optimal coverage (Nemhauser et al. 1978). │
│  Output: SelectedEvidence[] with marginal_gain scores.                    │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 9: Conformal Prediction Gate                                         │
│  Each selected chunk is scored for non-conformity against the calibration  │
│  set for this corpus. The threshold q̂ is the (1-α) quantile of the        │
│  calibration non-conformity scores where α = 1 - coverage_target.         │
│                                                                             │
│  Chunks with nonconformity_score > q̂ are REJECTED.                        │
│  If all chunks are rejected → Decision::Abstained.                        │
│  If ≥1 chunk passes → Decision::Answered.                                 │
│                                                                             │
│  Guarantee: P(true answer in retained set) ≥ coverage_target              │
│  (finite-sample, distribution-free — Vovk et al. 2005).                  │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 10: LLM Answer Generation + Audit Block                             │
│  If Decision::Answered: prompt the configured LLM with accepted evidence. │
│  The prompt includes each chunk with its source citation for grounding.   │
│  The answer is stored in EvidencePacket.answer.                           │
│                                                                             │
│  The AuditBlock is assembled: SHA-256 tamper-evident hash of              │
│  (query_id || query || evidence_ids || answer || model || timestamp).     │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
EvidencePacket (returned to caller)
```

---

## Crate Structure

Sievrant is a Cargo workspace of 15 crates. Each crate has a single responsibility.

| Crate | Responsibility |
|---|---|
| `sievrant-core` | Shared types: `EvidencePacket`, `EvidenceChunk`, `RetrievalConfig`, all enums |
| `sievrant-math` | RRF, Submodular selection (Lazy Greedy), Conformal quantile (Fenwick Tree) |
| `sievrant-index` | `VectorStore` trait + Qdrant and Memory implementations. ACL filtering logic. |
| `sievrant-rerank` | Cross-encoder reranking via ONNX local model or Cohere/Jina API |
| `sievrant-llm` | LLM provider trait + OpenAI, Anthropic, Gemini clients |
| `sievrant-query` | Query processing: expansion, HyDE, multi-hop agentic loop |
| `sievrant-orchestrator` | The 10-stage pipeline. Wires all crates together. |
| `sievrant-storage` | WAL, disk-backed chunk storage, LSM merge engine |
| `sievrant-acl` | ACL group resolution, permission validation |
| `sievrant-calibration` | Conformal calibration set management, threshold computation |
| `sievrant-audit` | Tamper-evident SHA-256 audit block generation |
| `sievrant-observability` | OpenTelemetry traces and metrics, coverage monitoring |
| `sievrant-server` | Axum REST server + Tonic gRPC server + MCP server |
| `sievrant-cli` | CLI for corpus management, calibration, diagnostics |
| `sievrant` | Public facade crate. Re-exports the stable public API. |

---

## Data Flow Types

The pipeline transforms data through these typed intermediate structures:

```
Query (String)
  └▶ RankedCandidate  { chunk_id, score, retriever, text, metadata: ChunkMetadata }
       └▶ FusedCandidate { chunk_id, fused_score, source_scores, text, metadata }
            └▶ RerankedCandidate { chunk_id, fused_score, rerank_score, text, metadata }
                 └▶ SelectedEvidence { chunk_id, fused_score, rerank_score, marginal_gain,
                 │                     nonconformity_score, text, metadata }
                 └▶ EvidenceChunk { chunk_id, source_doc, text, rrf_score, rerank_score,
                                     nonconformity_score, submodular_marginal_gain,
                                     acl_groups, freshness_hours, token_count, ... }
                      └▶ EvidencePacket { query_id, answer, decision, evidence,
                                           audit, coverage_guarantee, latency_ms, ... }
```

Every field has a named consumer. Nothing is discarded silently.

---

## Concurrency Model

Sievrant uses Tokio for async I/O throughout. Key concurrency decisions:

- **Stage 5 (index search)**: `tokio::task::JoinSet` spawns all 4 index queries concurrently. Total latency is max(index latencies), not sum.
- **Stage 7 (reranking)**: Batched ONNX inference on a Rayon thread pool to avoid blocking the Tokio executor.
- **Stage 10 (LLM)**: Single async call. No parallelism needed as LLM is typically the bottleneck.
- **Server layer**: Axum REST and Tonic gRPC run as separate `tokio::task::spawn` tasks sharing `Arc<AppState>`.
- **Write path**: WAL writes go through a crossbeam channel to a dedicated background flush thread.

---

## Deployment Modes

| Mode | How | When to use |
|---|---|---|
| Embedded library | `use sievrant::Corpus` in your Rust app | Rust services wanting zero network hop |
| Docker container | `docker-compose up` | Fastest path for Python/Node/any language |
| Kubernetes | Helm chart (see `deploy/helm/`) | Production multi-replica deployments |
| MCP server | `sievrant mcp` | Claude Desktop, Cursor, other MCP clients |
