# Sievrant Architecture

This document provides a technical overview of Sievrant's internal architecture, crate dependency graph, and data flow through the retrieval pipeline.

## Crate Dependency Graph

The workspace is organized into 17 crates, each with a single responsibility.
Dependencies flow downward: leaf crates at the top, foundational crates at the bottom.

```mermaid
graph TD
    SERVER["sievrant-server<br/>REST + gRPC + MCP"]
    CLI["sievrant-cli<br/>Command line tool"]
    FACADE["sievrant<br/>Public facade"]
    PYTHON["sievrant-python<br/>PyO3 bindings"]
    NODE["sievrant-node<br/>NAPI bindings"]
    ORCH["sievrant-orchestrator<br/>10-stage pipeline"]
    QUERY["sievrant-query<br/>Router, cache, processing"]
    INDEX["sievrant-index<br/>Vector, BM25, tree, graph"]
    RERANK["sievrant-rerank<br/>Cross-encoder reranking"]
    LLM["sievrant-llm<br/>LLM provider clients"]
    MATH["sievrant-math<br/>RRF, submodular, conformal"]
    STORAGE["sievrant-storage<br/>WAL, LSM merge"]
    ACL["sievrant-acl<br/>Access control"]
    CALIB["sievrant-calibration<br/>Conformal calibration"]
    AUDIT["sievrant-audit<br/>Tamper evident audit"]
    OBS["sievrant-observability<br/>Tracing, metering"]
    CORE["sievrant-core<br/>Types, config, errors"]

    SERVER --> ORCH
    SERVER --> CORE
    CLI --> FACADE
    FACADE --> ORCH
    PYTHON --> FACADE
    NODE --> FACADE
    ORCH --> QUERY
    ORCH --> INDEX
    ORCH --> RERANK
    ORCH --> LLM
    ORCH --> MATH
    ORCH --> STORAGE
    ORCH --> ACL
    ORCH --> CALIB
    ORCH --> AUDIT
    ORCH --> OBS
    QUERY --> CORE
    QUERY --> LLM
    INDEX --> CORE
    RERANK --> CORE
    LLM --> CORE
    MATH --> CORE
    STORAGE --> CORE
    ACL --> CORE
    CALIB --> CORE
    AUDIT --> CORE
    OBS --> CORE
```

## Data Flow: Single Query

The following sequence diagram shows how a single query flows through all 10 stages of the retrieval pipeline. Each stage either transforms or filters the data before passing it to the next.

```mermaid
sequenceDiagram
    participant Client
    participant Server
    participant Cache as Semantic Cache
    participant Router as Query Router
    participant QP as Query Processor
    participant ACL as ACL Resolver
    participant Idx as Indexes (4-way)
    participant RRF as RRF Fusion
    participant RR as Reranker
    participant Sub as Submodular Selector
    participant Conf as Conformal Gate
    participant LLM as LLM Provider

    Client->>Server: POST /v1/query {query, corpus}
    Server->>Cache: Stage 1: Lookup (query, corpus, user_id)
    alt Cache hit
        Cache-->>Server: Cached EvidencePacket
        Server-->>Client: Response (cache_hit: true)
    else Cache miss
        Cache-->>Server: None
        Server->>Router: Stage 2: Classify route
        Router-->>Server: Fast | Standard | Slow
        Server->>QP: Stage 3: Process query (expand, HyDE, decompose)
        QP-->>Server: ProcessedQuery (embeddings, sub-queries)
        Server->>ACL: Stage 4: Resolve user ACL groups
        ACL-->>Server: Vec of allowed groups
        Server->>Idx: Stage 5: Parallel search (4 indexes)
        Note over Idx: Vector, BM25, Tree, Graph searched concurrently
        Idx-->>Server: 4 ranked candidate lists
        Server->>RRF: Stage 6: Reciprocal Rank Fusion
        RRF-->>Server: Fused + deduplicated candidates
        Server->>RR: Stage 7: Cross-encoder reranking (top N)
        RR-->>Server: Reranked candidates with CE scores
        Server->>Sub: Stage 8: Submodular selection under token budget
        Sub-->>Server: Selected evidence set
        Server->>Conf: Stage 9: Conformal gate (coverage check)
        alt Passes gate
            Conf-->>Server: Decision: ANSWERED
            Server->>LLM: Stage 10: Generate answer from evidence
            LLM-->>Server: Answer text
        else Fails gate
            Conf-->>Server: Decision: ABSTAINED
        end
        Server-->>Client: EvidencePacket (answer, evidence, audit, decision)
    end
```

## Mathematical Foundations

Sievrant replaces LLM heuristics with mathematically grounded decision mechanisms at three critical points in the pipeline.

### Reciprocal Rank Fusion (Stage 6)

RRF combines ranked lists from heterogeneous retrieval systems without requiring score calibration. For a document d appearing at rank r in list L, the fused score is:

```
RRF(d) = sum over all lists L of: 1 / (k + rank_L(d))
```

where k is a constant (default 60, per Cormack et al. 2009). This is calibration-free: it works regardless of whether the underlying scores are probabilities, cosine similarities, or BM25 values.

### Submodular Evidence Selection (Stage 8)

Given a token budget B, selecting the optimal subset of evidence chunks is NP-hard. Sievrant uses the greedy algorithm for monotone submodular maximization, which provides a (1 minus 1/e) approximation guarantee (Nemhauser et al. 1978). The objective function balances relevance (reranker score) against redundancy (pairwise similarity between selected chunks).

### Conformal Prediction Gate (Stage 9)

The conformal gate provides distribution-free finite-sample coverage guarantees. Given a calibration set and a target coverage level alpha (default 0.95), the gate computes a threshold such that at least alpha fraction of future correct answers will pass the gate. This guarantee holds regardless of the underlying score distribution, requiring only the exchangeability assumption.

## Deployment Topology

```mermaid
graph LR
    subgraph "Client Layer"
        PY["Python SDK"]
        TS["Node.js SDK"]
        REST["REST Client"]
        GRPC["gRPC Client"]
        MCP["MCP / Claude"]
    end

    subgraph "Sievrant Server"
        API["HTTP/gRPC API<br/>Auth + Rate Limit"]
        PIPE["10-Stage Pipeline"]
    end

    subgraph "Backing Services"
        QD["Qdrant<br/>Vector Store"]
        LLM_SVC["LLM Provider<br/>OpenAI / Anthropic / Gemini"]
        JAEGER["Jaeger<br/>Distributed Tracing"]
    end

    PY --> API
    TS --> API
    REST --> API
    GRPC --> API
    MCP --> API
    API --> PIPE
    PIPE --> QD
    PIPE --> LLM_SVC
    PIPE --> JAEGER
```
