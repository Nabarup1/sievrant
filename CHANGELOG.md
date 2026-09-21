# CHANGELOG

All notable changes to Sievrant are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

**Core Engine**
- `sievrant-server`: Refactored `rest.rs` to extract DTOs into `dto.rs`.
- `sievrant-orchestrator`: Refactored `pipeline.rs` to extract functional stages into `stages.rs`.
- `sievrant-core`: Added `testing` module with mock data builders to facilitate offline integration testing.
- `sievrant-math`: Added integration tests for conformal gating boundary conditions.
- `sievrant-orchestrator`: Added mock implementations for `RetrievalIndex` and `LlmProvider` to enable end-to-end pipeline testing without Docker or external network calls.
- `.github/workflows/ci.yml`: Increased `cargo llvm-cov` threshold to 75%.
- `README.md`: Documented how to run `cargo test --workspace` for offline integration tests.

## [0.1.0] - 2026-09-09

### Added

**Core Engine**
- `sievrant-core`: Shared types: `EvidencePacket`, `EvidenceChunk`, `ChunkMetadata`, `RetrievalConfig`, `Decision`, `Route`, `RetrieverType`
- `sievrant-math`: Reciprocal Rank Fusion (RRF), Submodular Greedy evidence selection with Fenwick Tree-backed conformal quantile computation
- `sievrant-index`: `VectorStore` plugin trait with `QdrantVectorStore` and `MemoryVectorStore` implementations; ACL filtering at pre-retrieval stage
- `sievrant-rerank`: Cross-encoder reranker supporting local ONNX models, Cohere Rerank API, and Jina Rerank API
- `sievrant-llm`: `LlmProvider` trait with OpenAI, Anthropic, and Gemini implementations
- `sievrant-query`: Query processing (expansion, HyDE, multi-hop agentic loop via `DefaultAgenticRetriever`)
- `sievrant-orchestrator`: 10-stage `RetrievalOrchestrator` pipeline with full latency instrumentation per stage
- `sievrant-storage`: WAL-backed chunk storage with LSM merge engine
- `sievrant-acl`: ACL group resolution and permission validation
- `sievrant-calibration`: Conformal calibration set management and per-corpus threshold computation
- `sievrant-audit`: Tamper-evident SHA-256 `AuditBlock` generation (EU AI Act Art. 12/13)
- `sievrant-observability`: OpenTelemetry traces and metrics, coverage monitoring, cost metering
- `sievrant`: Public facade crate with stable `Corpus` API

**Server**
- `sievrant-server`: Dual-server: Axum REST API (port 8080) and Tonic gRPC API (port 50051)
- REST endpoints: `GET /v1/health`, `POST /v1/query`, `POST /v1/query/full`, `POST /v1/ingest`, `GET /v1/corpora`, `GET /v1/corpora/{corpus}`, `DELETE /v1/corpora/{corpus}/documents/{doc_id}`
- Swagger UI at `/swagger-ui`, OpenAPI JSON at `/api-docs/openapi.json`
- SHA-256 Bearer token authentication with multi-tenant isolation
- MCP server for Claude Desktop and Cursor integration

**CLI**
- `sievrant-cli`: `sievrant ingest`, `sievrant query`, `sievrant calibrate`, `sievrant server` subcommands

**Python SDK** (`sdks/python`)
- `SievrantClient` async httpx client
- `EvidencePacket`, `EvidenceChunk`, `AuditBlock`, `Decision`, `Route`, `RetrieverType` Pydantic models
- LangChain `SievrantRetriever` (BaseRetriever)
- LlamaIndex `SievrantRetriever` (BaseRetriever) and `SievrantQueryEngine` (BaseQueryEngine)
- DSPy `SievrantRM` (dspy.Retrieve)
- Full test suite with `pytest-httpx` mock transport

**Node.js SDK** (`sdks/nodejs`)
- `SievrantClient` TypeScript client (native fetch, Node 18+)
- Full TypeScript type definitions for all Sievrant types
- LangChain.js `SievrantRetriever`
- Vercel AI SDK `SievrantRetriever`
- ESM and CJS dual build

**Documentation**
- `docs/index.md`: Documentation landing page
- `docs/quickstart.md`: 5-minute getting started guide (Docker, Python, Node.js, Rust)
- `docs/concepts/architecture.md`: Full 10-stage pipeline with Mermaid diagram and crate map
- `docs/concepts/conformal-prediction.md`: Mathematical deep dive with academic references
- `docs/concepts/submodular-selection.md`: Submodular optimization theory and implementation
- `docs/concepts/evidence-packet.md`: Complete field reference with usage examples
- `docs/guides/installation.md`: All installation methods
- `docs/guides/python-sdk.md`: Full Python SDK guide
- `docs/guides/nodejs-sdk.md`: Full Node.js SDK guide
- `docs/guides/langchain-integration.md`: LangChain drop-in replacement guide
- `docs/api-reference/rest-api.md`: Full REST API reference with examples
- `docs/comparison/vs-langchain.md`: Honest technical comparison

**Infrastructure**
- `docker-compose.yml`: Sievrant server + Qdrant + Jaeger stack
- `Dockerfile.backend`: Multi-stage Rust build
- `Makefile`: Development workflow commands
- `.env.example`: All configuration options documented

[0.1.0]: https://github.com/Nabarup1/sievrant/releases/tag/v0.1.0
