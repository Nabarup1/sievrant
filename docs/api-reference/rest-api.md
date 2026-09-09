# REST API Reference

Sievrant exposes a REST API that follows standard HTTP conventions. All endpoints accept and return JSON. An interactive Swagger UI is available at `/swagger-ui` on any running Sievrant server.

**Base URL**: `http://your-sievrant-server:8080`

---

## Authentication

All non-health endpoints require a Bearer token:

```
Authorization: Bearer sk-your-api-key
```

Invalid or missing tokens return:

```json
HTTP 401 Unauthorized
{
  "error": "unauthorized",
  "message": "Missing or invalid Authorization header"
}
```

---

## Endpoints

### `GET /v1/health`

Check server health. No authentication required.

**Response** `200 OK`:
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

---

### `POST /v1/query`

Execute a retrieval query against a corpus.

**Request body**:

```json
{
  "query": "What is the monthly rent amount?",
  "corpus": "legal-contracts",
  "coverage": 0.95,
  "token_budget": 4000,
  "acl_groups": ["legal", "finance"],
  "skip_cache": false,
  "route": "AUTO"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | `string` | Yes | — | Natural language query |
| `corpus` | `string` | Yes | — | Corpus name to search |
| `coverage` | `float` | No | `0.95` | Conformal coverage target (0.0 to 1.0) |
| `token_budget` | `integer` | No | `4000` | Max total tokens across evidence chunks |
| `acl_groups` | `string[]` | No | `[]` | ACL groups to filter to (caller's groups) |
| `skip_cache` | `boolean` | No | `false` | Skip the semantic cache |
| `route` | `string` | No | `"AUTO"` | `"AUTO"`, `"FAST"`, `"STANDARD"`, `"SLOW"` |

**Response** `200 OK`:

```json
{
  "query_id": "01932a7b-8f5c-7a3d-b2e1-4f9c8a3d7e6f",
  "decision": "ANSWERED",
  "answer": "The monthly rent is $4,500 USD.",
  "evidence_count": 3,
  "coverage_guarantee": 0.95,
  "latency_ms": 47,
  "cache_hit": false,
  "route": "FAST"
}
```

> Note: The `query` endpoint returns a summary response. To get the full `EvidencePacket` with all chunks and the audit block, use `POST /v1/query/full`.

**Response** `401 Unauthorized` — invalid token

**Response** `404 Not Found` — corpus does not exist

**Response** `500 Internal Server Error`

---

### `POST /v1/query/full`

Same as `/v1/query` but returns the complete `EvidencePacket` including all evidence chunks, per-stage latency breakdown, and the tamper-evident audit block.

**Response** `200 OK`:

```json
{
  "query_id": "01932a7b-...",
  "trace_id": "4bf92f3577b34da6...",
  "decision": "ANSWERED",
  "answer": "The monthly rent is $4,500 USD.",
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
      "path": ["Section 2", "Payment Terms"],
      "page": 3,
      "char_span": [1882, 2043],
      "text": "Monthly rent is $4,500 USD...",
      "retriever": "vector_index",
      "rrf_score": 0.0164,
      "rerank_score": 0.92,
      "nonconformity_score": 0.08,
      "submodular_marginal_gain": 0.41,
      "ingested_at": "2025-01-15T08:00:00Z",
      "index_generation": 1,
      "acl_groups": ["legal"],
      "freshness_hours": 48.2,
      "token_count": 18
    }
  ],
  "audit": {
    "model_version": "gpt-4o-2024-08-06",
    "retrieval_stages_ms": {
      "cache_check": 1, "routing": 2, "query_processing": 4,
      "acl_resolution": 1, "vector_search": 18, "bm25_search": 6,
      "tree_search": 4, "graph_search": 8, "rrf_fusion": 2,
      "reranking": 21, "submodular_selection": 3, "conformal_gate": 1,
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

### `POST /v1/ingest`

Ingest documents into a corpus. The corpus is created automatically if it does not exist.

**Request body**:

```json
{
  "corpus": "legal-contracts",
  "documents": [
    {
      "id": "lease-001",
      "content": "The lease term is 60 months starting January 1, 2025...",
      "metadata": {
        "filename": "lease.pdf",
        "author": "Legal Team",
        "created_at": "2025-01-01T00:00:00Z",
        "custom_field": "any value"
      },
      "acl_groups": ["legal", "finance"]
    }
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `corpus` | `string` | Yes | Target corpus name |
| `documents` | `Document[]` | Yes | Array of documents to ingest |
| `documents[].id` | `string` | Yes | Unique document ID within the corpus |
| `documents[].content` | `string` | Yes | Full text content |
| `documents[].metadata` | `object` | No | Arbitrary key-value metadata |
| `documents[].acl_groups` | `string[]` | No | Access control groups |

**Response** `202 Accepted`:

```json
{
  "corpus": "legal-contracts",
  "ingested": 1,
  "failed": 0
}
```

---

### `GET /v1/corpora`

List all registered corpora visible to the authenticated tenant.

**Response** `200 OK`:

```json
{
  "corpora": [
    {
      "name": "legal-contracts",
      "created_at": "2025-01-15T08:00:00Z",
      "document_count": 2847,
      "chunk_count": 18923,
      "index_generation": 3,
      "calibration_set_id": "contracts-v2"
    }
  ]
}
```

---

### `GET /v1/corpora/{corpus}`

Get metadata for a specific corpus.

**Response** `200 OK`: Same as a single item in `GET /v1/corpora`

**Response** `404 Not Found` — corpus not found

---

### `DELETE /v1/corpora/{corpus}/documents/{doc_id}`

Delete a specific document and all its chunks from a corpus.

**Response** `200 OK`:

```json
{"deleted": true, "corpus": "legal-contracts", "doc_id": "lease-001"}
```

---

## Rate Limits

Default rate limits per API key:

| Endpoint | Limit |
|---|---|
| `POST /v1/query` | 1000 req/min |
| `POST /v1/ingest` | 100 req/min |
| All other | 500 req/min |

Rate limit headers are included in every response:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1733123456
```

When exceeded: `HTTP 429 Too Many Requests`

---

## OpenAPI / Swagger

Every running Sievrant server exposes:

- Interactive UI: `GET /swagger-ui`
- Raw OpenAPI JSON: `GET /api-docs/openapi.json`

---

## Error Response Format

All errors follow the same shape:

```json
{
  "error": "error_code",
  "message": "Human readable description",
  "request_id": "01932a7b-..."
}
```

| HTTP Code | `error` value | Cause |
|---|---|---|
| `400` | `bad_request` | Invalid JSON or missing required field |
| `401` | `unauthorized` | Missing or invalid API key |
| `404` | `corpus_not_found` | Corpus does not exist |
| `422` | `validation_error` | Field value out of range (e.g., coverage > 1.0) |
| `429` | `rate_limited` | Rate limit exceeded |
| `500` | `internal_error` | Server-side error |
| `503` | `service_unavailable` | Underlying index or model temporarily unavailable |
