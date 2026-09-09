# Rust SDK Guide

Sievrant ships as a Rust workspace. This guide covers two distinct use cases:

1. **Embedding the facade library crate** (`sievrant` on crates.io) to use Sievrant types and the fluent builder API inside a Rust application that connects to a running Sievrant server over HTTP or gRPC.
2. **Running the server binary** (`sievrant-server`) as a standalone REST and gRPC service that your Python, Node.js, or browser clients can query.

> [!NOTE]
> The `sievrant` library crate provides the type definitions, the `RetrieveBuilder` fluent API, and SDK-level conveniences. The `Corpus::create()` and `Corpus::open()` entry points require a running backend to wire up to. For direct embedding without a server, contact us for the self-hosted backend integration guide.

---

## Installation

### Running the server binary

The fastest path is running the pre-built Docker image or compiling the server yourself:

```bash
git clone https://github.com/Nabarup1/sievrant
cd sievrant
cp .env.example .env           # set OPENAI_API_KEY and other provider keys
docker-compose up -d           # starts Qdrant + sievrant-server + OpenTelemetry
```

Or compile from source:

```bash
cargo install --path crates/sievrant-server
sievrant-server
```

The server starts on:
- REST: `http://0.0.0.0:8080`
- gRPC: `0.0.0.0:50051`

### Library dependency

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
sievrant = "0.1"
tokio = { version = "1", features = ["full"] }
```

---

## Core Concepts

### The Evidence Packet

Every query returns an `EvidencePacket`. This is the central data structure of Sievrant:

```rust
use sievrant::{EvidencePacket, Decision, Route};

let packet: EvidencePacket = /* result of retrieve() */;

match packet.decision {
    Decision::Answered => {
        println!("Answer: {:?}", packet.answer);
        println!("Confidence guaranteed: {}%", packet.coverage_guarantee * 100.0);
        println!("Route taken: {:?}", packet.route);     // Fast, Standard, Slow, or Cached
        println!("Latency: {}ms", packet.latency_ms);
    }
    Decision::Abstained => {
        // Sievrant found no evidence meeting the coverage guarantee.
        // Your app should handle this gracefully.
        println!("Insufficient evidence. Audit: {:?}", packet.audit.session_hash);
    }
    Decision::EvidenceOnly => {
        // Called without LLM generation (evidence-only mode)
        println!("Evidence chunks: {}", packet.evidence.len());
    }
}
```

### Evidence Chunks

Each chunk in `packet.evidence` carries full provenance:

```rust
for chunk in &packet.evidence {
    println!("Chunk ID:    {}", chunk.chunk_id);
    println!("Source doc:  {}", chunk.source_doc);
    println!("Path:        {}", chunk.path.join(" > "));
    println!("Page:        {:?}", chunk.page);
    println!("RRF score:   {:.4}", chunk.rrf_score);
    println!("Rerank:      {:?}", chunk.rerank_score);
    println!("NCS:         {:.4}", chunk.nonconformity_score);
    println!("Tokens:      {}", chunk.token_count);
    println!("ACL groups:  {:?}", chunk.acl_groups);
    println!("---");
}
```

### The Audit Block

Every `EvidencePacket` includes an `AuditBlock` for compliance and tamper detection:

```rust
let audit = &packet.audit;
println!("Session hash:      {}", audit.session_hash);
println!("Corpus hash:       {}", audit.corpus_hash);
println!("Model used:        {:?}", audit.model_used);
println!("Retrieval stages:  {:?}", audit.stages_ms);
```

The `session_hash` is a SHA-256 over the query, evidence set, and timestamp. If any chunk is altered after the fact, the hash will not match. This is your EU AI Act audit trail.

---

## Fluent Builder API

The `RetrieveBuilder` provides fine-grained control over every pipeline parameter:

```rust
use std::sync::Arc;
use sievrant::{RetrieveBuilder, RouteStrategy};

let packet = RetrieveBuilder::new("What is the termination clause?", &corpus)
    .coverage(0.99)               // Require 99% statistical coverage (more conservative)
    .budget(2000)                 // Maximum 2000 tokens across all evidence chunks
    .route(RouteStrategy::Slow)   // Force multi-hop decomposition
    .user("user-uuid-abc123")     // Tenant-isolated cache and ACL resolution
    .skip_cache(false)            // Allow semantic cache hits (default)
    .await_result()
    .await?;
```

### Available Route Strategies

| Strategy | What It Does | When to Use |
|---|---|---|
| `RouteStrategy::Auto` | Adaptive router decides based on query complexity | Default. Use for most production traffic. |
| `RouteStrategy::Fast` | Single-pass hybrid search, no expansion | Simple lookups, latency-sensitive APIs |
| `RouteStrategy::Standard` | Full pipeline with query expansion | Most complex questions |
| `RouteStrategy::Slow` | Multi-hop decomposition with iterative sub-queries | Cross-document reasoning, comparisons |

---

## Ingestion

Documents are ingested via the REST API. From Rust you can call the server HTTP endpoint directly or use the CLI:

```bash
cargo run -p sievrant-cli -- --token sk-your-key ingest \
  --corpus contracts \
  --file lease.pdf
```

Or via `reqwest` in your Rust application:

```rust
use reqwest::Client;
use serde_json::json;

let client = Client::new();
let resp = client
    .post("http://localhost:8080/v1/ingest")
    .bearer_auth("sk-your-key")
    .json(&json!({
        "corpus": "contracts",
        "documents": [
            {
                "id": "lease-001",
                "content": "The lease commences January 1, 2025 for 60 months.",
                "metadata": {
                    "filename": "lease.pdf",
                    "department": "legal"
                }
            }
        ]
    }))
    .send()
    .await?;

println!("Status: {}", resp.status());
println!("Body: {}", resp.text().await?);
```

---

## gRPC API

Sievrant exposes a gRPC service defined in `proto/sievrant.proto`. You can call it from any gRPC client. From Rust using `tonic`:

```toml
[dependencies]
tonic = "0.12"
prost = "0.13"
tokio = { version = "1", features = ["full"] }
```

```rust
use tonic::transport::Channel;
use tonic::metadata::MetadataValue;

// The generated client from sievrant.proto
// use sievrant_server::api::grpc::proto::sievrant_service_client::SievrantServiceClient;

let channel = Channel::from_static("http://localhost:50051")
    .connect()
    .await?;

// Attach bearer token as gRPC metadata
let token: MetadataValue<_> = "Bearer sk-your-key".parse()?;
// ... add interceptor to attach token header on every call
```

---

## Configuration Reference

The server is configured via environment variables or a `.env` file. Copy `.env.example` to `.env` and edit:

```bash
# LLM Provider (at least one required for answer generation)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Vector Database
SIEVRANT_QDRANT_URL=http://localhost:6333

# Server
SIEVRANT_REST_ADDR=0.0.0.0:8080
SIEVRANT_GRPC_ADDR=0.0.0.0:50051

# Observability
RUST_LOG=sievrant=info,tower_http=debug
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Conformal Prediction Defaults
SIEVRANT_DEFAULT_COVERAGE=0.95
SIEVRANT_DEFAULT_TOKEN_BUDGET=4000
```

---

## CLI Reference

The `sievrant-cli` binary provides a terminal interface to any running Sievrant server:

```bash
# Install
cargo install --path crates/sievrant-cli

# Health check
sievrant --token sk-your-key health

# Query a corpus
sievrant --token sk-your-key query \
  --corpus contracts \
  "What is the penalty for early termination?"

# Point at a non-default server
sievrant --url http://prod.company.com:8080 \
  --token sk-your-key \
  health
```

Run `sievrant --help` for the full command reference.

---

## Error Handling

All fallible operations return `sievrant::Result<T>`, which is an alias for `Result<T, SievrantError>`:

```rust
use sievrant::{SievrantError, retrieve};

match retrieve("my query", &corpus).await {
    Ok(packet) => handle_packet(packet),
    Err(SievrantError::Storage(e)) => eprintln!("Storage error: {e}"),
    Err(SievrantError::Llm(e)) => eprintln!("LLM error: {e}"),
    Err(SievrantError::NotFound(msg)) => eprintln!("Not found: {msg}"),
    Err(e) => eprintln!("Unexpected error: {e}"),
}
```

---

## Running Tests

```bash
# Run the entire test suite (144 tests across all crates)
cargo test --workspace

# Run only mathematical correctness tests (fast, no I/O)
cargo test -p sievrant-math

# Run property-based tests with more iterations
PROPTEST_CASES=1000 cargo test -p sievrant-math

# Run server integration tests (requires no external services)
cargo test -p sievrant-server
```

---

## Building for Production

```bash
# Optimized release binary
cargo build --release --bin sievrant-server

# The binary is at:
./target/release/sievrant-server

# Or use Docker (recommended for production)
docker build -f Dockerfile.backend -t sievrant-server .
docker run -p 8080:8080 -e OPENAI_API_KEY=sk-... sievrant-server
```

The Dockerfile uses `cargo-chef` for aggressive layer caching. Only changed crates are recompiled, keeping CI rebuild times short.

---

## Next Steps

- [Architecture deep dive](../concepts/architecture.md)
- [Conformal Prediction explained](../concepts/conformal-prediction.md)
- [Docker deployment guide](docker-compose.md)
- [REST API reference](../api-reference/rest-api.md)
- [Python SDK guide](python-sdk.md)
