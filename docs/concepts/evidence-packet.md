# Evidence Packet

The `EvidencePacket` is the primary output of every Sievrant retrieval call. It is both the API response and the compliance artifact under EU AI Act Articles 12 and 13.

Every field has a named consumer. Nothing is included for cosmetic reasons.

---

## Full Structure

```json
{
  "query_id": "01932a7b-8f5c-7a3d-b2e1-4f9c8a3d7e6f",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "decision": "ANSWERED",
  "answer": "The monthly rent is $4,500 USD, payable on the first of each month.",
  "coverage_guarantee": 0.95,
  "empirical_coverage_last_30d": 0.962,
  "route": "FAST",
  "cache_hit": false,
  "latency_ms": 47,
  "cost_usd": 0.0021,
  "evidence": [
    {
      "chunk_id": "lease-001:sec_2:p1",
      "source_doc": "lease.pdf",
      "doc_version_hash": "sha256:a3f8b2...",
      "path": ["Section 2", "Payment Terms", "paragraph 1"],
      "page": 3,
      "char_span": [1882, 2043],
      "text": "Monthly rent is $4,500 USD, payable on the first of each month.",
      "retriever": "vector_index",
      "rrf_score": 0.0164,
      "rerank_score": 0.92,
      "nonconformity_score": 0.08,
      "submodular_marginal_gain": 0.41,
      "ingested_at": "2025-01-15T08:00:00Z",
      "index_generation": 1,
      "acl_groups": ["legal", "finance"],
      "freshness_hours": 48.2,
      "token_count": 18
    }
  ],
  "audit": {
    "model_version": "gpt-4o-2024-08-06",
    "retrieval_stages_ms": {
      "cache_check": 1,
      "routing": 2,
      "query_processing": 4,
      "acl_resolution": 1,
      "vector_search": 18,
      "bm25_search": 6,
      "tree_search": 4,
      "graph_search": 8,
      "rrf_fusion": 2,
      "reranking": 21,
      "submodular_selection": 3,
      "conformal_gate": 1,
      "llm_generation": 340
    },
    "total_candidates_before_rerank": 100,
    "total_candidates_after_rerank": 25,
    "total_candidates_after_conformal": 4,
    "calibration_set_id": "contracts-v2",
    "session_hash": "sha256:7f83b1657..."
  }
}
```

---

## Field Reference

### Top-Level Fields

| Field | Type | Description |
|---|---|---|
| `query_id` | `string` (UUID v4) | Unique ID for this retrieval. Ties logs, traces, and audit records together. |
| `trace_id` | `string?` | OpenTelemetry trace ID. Present when OTel is configured. Links to Jaeger/Tempo. |
| `decision` | `enum` | `ANSWERED`, `ABSTAINED`, or `EVIDENCE_ONLY`. Drives agent branching. |
| `answer` | `string?` | Generated answer. `null` when decision is `ABSTAINED` or `EVIDENCE_ONLY`. |
| `coverage_guarantee` | `float` | The configured coverage target (e.g., 0.95). |
| `empirical_coverage_last_30d` | `float?` | Actual observed coverage over the last 30 days for this corpus. `null` until enough data accumulates. |
| `route` | `enum` | `FAST`, `STANDARD`, `SLOW`, or `CACHED`. Which pipeline path this query took. |
| `cache_hit` | `bool` | `true` when this packet was served from the semantic cache without running the pipeline. |
| `latency_ms` | `integer` | Total wall-clock latency from query receipt to packet serialization. |
| `cost_usd` | `float?` | Estimated USD cost (embedding + reranker + LLM). `null` when using local models. |

### Decision Values

| Value | Meaning | Agent action |
|---|---|---|
| `ANSWERED` | Evidence passed the conformal gate. True answer is in evidence with P ≥ coverage. | Render the answer with citations. |
| `ABSTAINED` | All evidence rejected by conformal gate. Cannot guarantee answer is grounded. | Show "I don't know" or escalate to human. |
| `EVIDENCE_ONLY` | Evidence retrieved but LLM generation was skipped (requested by caller). | Show evidence directly to user or agent. |

### EvidenceChunk Fields

| Field | Type | Description |
|---|---|---|
| `chunk_id` | `string` | Globally unique: `"doc_id:section_idx:chunk_idx"` |
| `source_doc` | `string` | Source document filename or URI. |
| `doc_version_hash` | `string` | SHA-256 of source document at ingestion. Detect stale sources. |
| `path` | `string[]` | Hierarchical path: `["Section 4", "4.2 Term", "paragraph 3"]` |
| `page` | `integer?` | Page number if the format supports pagination. |
| `char_span` | `[int, int]?` | Character span `[start, end]` in the source for inline highlighting. |
| `text` | `string` | The actual text content of this chunk. |
| `retriever` | `enum` | Which index surfaced this chunk: `vector_index`, `bm25_index`, `tree_index`, `graph_index` |
| `rrf_score` | `float` | Reciprocal Rank Fusion score. Always positive. |
| `rerank_score` | `float?` | Cross-encoder score. `null` if reranking was skipped (degraded mode). |
| `nonconformity_score` | `float` | Conformal non-conformity. Lower = more trustworthy. |
| `submodular_marginal_gain` | `float` | Marginal coverage gain assigned during selection. |
| `ingested_at` | `ISO 8601` | When this chunk was written to the corpus. |
| `index_generation` | `integer` | LSM merge generation at indexing time. Detect stale results from cached indexes. |
| `acl_groups` | `string[]` | Access control groups this chunk belongs to. |
| `freshness_hours` | `float` | Hours since ingestion. Use to filter stale evidence in time-sensitive domains. |
| `token_count` | `integer` | Approximate token count (for budget accounting). |

### AuditBlock Fields

| Field | Type | Description |
|---|---|---|
| `model_version` | `string?` | LLM model identifier, e.g. `"gpt-4o-2024-08-06"` |
| `retrieval_stages_ms` | `object` | Per-stage wall-clock latency breakdown (13 stages). |
| `total_candidates_before_rerank` | `integer` | Candidate count entering the reranker (post-RRF). |
| `total_candidates_after_rerank` | `integer` | Candidate count after reranking, before submodular selection. |
| `total_candidates_after_conformal` | `integer` | Evidence count that passed the conformal gate. |
| `calibration_set_id` | `string` | Identifies the calibration set used. `"default"` when no per-corpus calibration exists. |
| `session_hash` | `string` | SHA-256 of `(query_id ║ query_text ║ evidence_ids ║ answer ║ model ║ timestamp)`. Tamper-evident. |

---

## Using EvidencePacket in Your Agent

### Branching on decision

```python
result = await client.query(corpus="contracts", query="What is the rent?")

if result.decision == "ANSWERED":
    # Safe to present answer
    print(f"Answer: {result.answer}")
    for chunk in result.evidence:
        print(f"Source: {chunk.source_doc} (p.{chunk.page}) — confidence: {1 - chunk.nonconformity_score:.2f}")

elif result.decision == "ABSTAINED":
    # Do NOT hallucinate. Escalate.
    print("Insufficient evidence. Please consult the source documents directly.")
    
elif result.decision == "EVIDENCE_ONLY":
    # Present evidence directly for human review
    for chunk in result.evidence:
        print(f"{chunk.text}\n  [Source: {chunk.source_doc}]")
```

### Audit log integration

```python
import json, hashlib

result = await client.query(corpus="contracts", query="What is the rent?")

# The session_hash can be independently verified
audit_input = "|".join([
    result.query_id,
    "What is the rent?",
    ",".join(c.chunk_id for c in result.evidence),
    result.answer or "",
    result.audit.model_version or "",
])
expected_hash = hashlib.sha256(audit_input.encode()).hexdigest()
assert result.audit.session_hash == f"sha256:{expected_hash}"
```
