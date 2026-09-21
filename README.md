# Sievrant

**The statistically audited retrieval decision engine for enterprise AI.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.78%2B-orange.svg)](https://www.rust-lang.org)
[![PyPI](https://img.shields.io/pypi/v/sievrant.svg)](https://pypi.org/project/sievrant)
[![npm](https://img.shields.io/npm/v/sievrant.svg)](https://www.npmjs.com/package/sievrant)

```
                   ┌────────────────────────────────────────────┐
  Your App  ──────▶│            Sievrant Server                 │──────▶  LLM
  REST/gRPC         │  Vector ─┐                                 │
  Python SDK        │  BM25  ──┼▶ RRF ─▶ Rerank ─▶ Submodular  │
  Node SDK          │  Graph ──┘          Math        Gate       │
                   │         ┌─────────────────────────────────┐│
                   │         │   Conformal Prediction (95% CI) ││
                   │         └─────────────────────────────────┘│
                   └────────────────────────────────────────────┘
```

Sievrant is the **calibrated decision layer** that sits between your vector database and your AI agents. It is not a vector database. It is not an agent framework. It is the layer that ensures your AI retrieval is **mathematically correct**.

---

## The Problem

Every production RAG system suffers from the same failures:

| Problem | How standard RAG handles it | Why it fails |
|---|---|---|
| Which index to trust | LLM "reasons" about tool calls | Non-deterministic, varies run to run |
| Which chunks go into context | Top-k by similarity score | Redundant chunks waste tokens |
| When to stop retrieving | LLM "feels" confident | No statistical basis, silent failures |
| Whether to trust a chunk | LLM eyeballs it | No formal error bound |

These are not LLM problems. They are **decision theory problems** with known, provable solutions.

---

## The Solution

Sievrant replaces every LLM guess with a closed-form or provably bounded operation:

| Decision | Sievrant mechanism | Mathematical basis |
|---|---|---|
| Which index to trust | Reciprocal Rank Fusion (RRF) | Rank-based, calibration-free (Cormack et al. 2009) |
| Which chunks make context | Submodular greedy under token budget | (1 - 1/e) approximation (Nemhauser et al. 1978) |
| When to stop retrieving | Conformal prediction coverage gate | Distribution-free finite-sample guarantee |
| Whether to trust a chunk | Conformalized retrieval scoring | Formally bounded P(off-source) |

The LLM's role shrinks to two places: understanding the query and writing the final answer. It never decides routing, selection, or stopping.

---

## Quick Start from a Fresh Clone

The following sequence is the complete path from zero to a running server with
no manual steps beyond filling in your API key:

```bash
# 1. Clone the repo
git clone https://github.com/Nabarup1/sievrant
cd sievrant

# 2. Copy the example env file and add your OpenAI key (or any compatible LLM key)
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-...

# 3. Start Qdrant and the Sievrant API server in the background
docker compose up -d

# 4. Verify the server is healthy
curl http://localhost:8080/v1/health
# Expected response: {"status":"ok","version":"0.1.0"}
```

That is it. The API is now reachable at `http://localhost:8080`.

---

## Quickstart

### Docker (30 seconds)

```bash
docker-compose up -d
curl -X POST http://localhost:8080/v1/query \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the lease term?", "corpus": "contracts"}'
```

### Python (pip install)

```bash
pip install sievrant
```

```python
import asyncio
from sievrant import SievrantClient

async def main():
    client = SievrantClient(base_url="http://localhost:8080", api_key="sk-your-key")

    # Ingest a document
    await client.ingest(
        corpus="contracts",
        documents=[{"id": "lease-001", "content": "The lease term is 60 months..."}]
    )

    # Query with a 95% statistical coverage guarantee
    result = await client.query(corpus="contracts", query="What is the lease term?")

    print(result.decision)       # "ANSWERED" or "ABSTAINED"
    print(result.answer)         # Generated answer, or None if abstained
    print(result.evidence)       # List of EvidenceChunk with scores, sources, ACL
    print(result.latency_ms)     # Total end-to-end latency
    print(result.audit)          # Full tamper-evident audit block

asyncio.run(main())
```

### Rust (native embed)

```rust
use sievrant::{Corpus, CorpusConfig, RetrievalConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let corpus = Corpus::new("contracts", CorpusConfig::default()).await?;

    corpus.ingest(vec![Document {
        id: "lease-001".into(),
        content: "The lease term is 60 months...".into(),
        ..Default::default()
    }]).await?;

    let packet = corpus.retrieve(
        "What is the lease term?",
        RetrievalConfig { coverage: 0.95, ..Default::default() }
    ).await?;

    println!("{:?}", packet.decision);   // Answered or Abstained
    println!("{:#?}", packet.evidence);  // Structured evidence with provenance
    Ok(())
}
```

### Node.js / TypeScript

```bash
npm install sievrant
```

```typescript
import { SievrantClient } from 'sievrant';

const client = new SievrantClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'sk-your-key',
});

const result = await client.query({
  corpus: 'contracts',
  query: 'What is the lease term?',
});

console.log(result.decision);     // "ANSWERED" or "ABSTAINED"
console.log(result.answer);       // Answer string or null
console.log(result.evidence);     // Typed EvidenceChunk[]
```

### LangChain drop-in

```python
from sievrant.integrations.langchain import SievrantRetriever

# Drop into any existing LangChain chain
retriever = SievrantRetriever(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="contracts"
)

# Works with RetrievalQA, ConversationalRetrievalChain, etc.
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

---

## Architecture

```
                    ┌────────────────────────────────────────────────────────┐
  Clients           │                     Sievrant Server                    │
  ─────────         │                                                        │
  REST API   ──────▶│  Auth Middleware ─▶ Tenant Isolation ─▶ Orchestrator  │
  gRPC       ──────▶│                                              │         │
  Python SDK        │              ┌───────────────────────────────┘         │
  Node SDK          │              ▼                                         │
  Rust embed        │  ┌──────────────────────────────────────────────────┐  │
  MCP / Claude      │  │                 10-Stage Pipeline                 │  │
                    │  │                                                   │  │
                    │  │  1. Semantic Cache Check (3-tier)                │  │
                    │  │  2. Adaptive Query Router (Fast/Standard/Slow)   │  │
                    │  │  3. Query Processing (expansion, HyDE, decomp.)  │  │
                    │  │  4. ACL Resolution (pre-retrieval filtering)     │  │
                    │  │  5. 4-way Parallel Index Search                  │  │
                    │  │     ├── Vector Index (Qdrant/Memory)             │  │
                    │  │     ├── BM25 Lexical Index (Tantivy)             │  │
                    │  │     ├── Hierarchical Tree Index                  │  │
                    │  │     └── Knowledge Graph (PPR)                   │  │
                    │  │  6. Reciprocal Rank Fusion (RRF)                 │  │
                    │  │  7. Cross-Encoder Reranking (ONNX / API)        │  │
                    │  │  8. Submodular Evidence Selection (token budget) │  │
                    │  │  9. Conformal Prediction Gate (coverage ≥ 0.95)  │  │
                    │  │  10. LLM Answer Generation + Audit Block         │  │
                    │  └──────────────────────────────────────────────────┘  │
                    └────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Sievrant | LangChain | LlamaIndex | Haystack |
|---|:---:|:---:|:---:|:---:|
| Statistical coverage guarantee | Yes | No | No | No |
| Hybrid 4-way retrieval | Yes | Partial | Partial | Partial |
| Submodular token budget | Yes | No | No | No |
| Conformal abstention | Yes | No | No | No |
| EU AI Act audit trail | Yes | No | No | No |
| Multi-tenant ACL | Yes | No | No | Partial |
| Built in Rust (zero GC) | Yes | No | No | No |
| gRPC + REST dual server | Yes | No | No | No |
| MCP native (Claude/Cursor) | Yes | No | No | No |
| LangChain compatible | Yes | Yes | No | No |
| LlamaIndex compatible | Yes | No | Yes | No |
| DSPy compatible | Yes | No | No | No |

---

## Documentation

Full documentation is at [docs/index.md](docs/index.md).

- [Quickstart](docs/quickstart.md)
- [Architecture deep dive](docs/concepts/architecture.md)
- [Conformal Prediction explained](docs/concepts/conformal-prediction.md)
- [Python SDK guide](docs/guides/python-sdk.md)
- [Node.js SDK guide](docs/guides/nodejs-sdk.md)
- [Rust SDK guide](docs/guides/rust-sdk.md)
- [LangChain integration](docs/guides/langchain-integration.md)
- [LlamaIndex integration](docs/guides/llama-index-integration.md)
- [Docker deployment](docs/guides/docker-compose.md)
- [REST API reference](docs/api-reference/rest-api.md)
- [vs LangChain](docs/comparison/vs-langchain.md)

---

## Benchmarks

Sievrant hybrid retrieval (Vector + BM25 + RRF + SIMD ColBERT Reranking) outperforms single-modality retrieval across our internal test datasets. Full benchmark methodology, reproducible scripts, and detailed results are in [docs/comparison/benchmarks.md](docs/comparison/benchmarks.md).

**Internal subset test (BEIR SciFact, 300 queries):**

| Metric | Sievrant Hybrid + Reranker | Dense vector only | BM25 only |
|---|---|---|---|
| Recall@100 | **0.85** | 0.67 | 0.61 |
| Median latency (ms) | 45 | 38 | 12 |
| p99 latency (ms) | 180 | 165 | 55 |

**Rust backend throughput (health endpoint, oha load test, 10k req @ 100 concurrency):**

| Metric | Result |
|---|---|
| Requests per second | ~33,400 |
| p99 latency | ~10.3 ms |
| Success rate | 100% |

> [!NOTE]
> Numbers above are from internal subset tests on a single WSL/Ubuntu developer machine. A reproducible benchmark script for independent verification is in `benchmarks/`. Full BEIR and MTEB suite results are planned.

---

## Installation

### Server (Docker)

```bash
git clone https://github.com/Nabarup1/sievrant
cd sievrant
cp .env.example .env   # add your OPENAI_API_KEY or similar
docker-compose up -d
```

### Server (Rust binary)

```bash
cargo install sievrant-server
sievrant-server --config config.toml
```

### Python client

```bash
pip install sievrant
```

### Node.js client

```bash
npm install sievrant
```

### Rust (embed in your Rust app)

```toml
[dependencies]
sievrant = "0.1"
```

## Offline Testing

The Sievrant workspace is built for completely offline, dependency-free integration testing using mock traits. You can verify pipeline logic without Docker, Qdrant, or any API keys.

```bash
# Run all workspace tests, including orchestrator decision paths, completely offline
cargo test --workspace
```

---

## Reproducing Benchmarks

All benchmark runs are seeded and produce a `config.json` alongside the results
so every run is fully reproducible.

```bash
# From a fresh clone, with the server running (see Quick Start above)
cd benchmarks

# Run BEIR SciFact with seed 42 (default)
bash run_beir.sh --dataset scifact --seed 42

# Results are written to benchmarks/results/scifact_YYYYMMDD.json
# along with a config.json recording the seed, model, and timestamp
```

For a manual run using the Python benchmark script directly:

```bash
source ../.venv/bin/activate
python beir_benchmark.py --dataset scifact --seed 42
```

See [docs/comparison/benchmarks.md](docs/comparison/benchmarks.md) for the
full methodology, hardware details, and comparison methodology.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome.

---

## Security

Please see [SECURITY.md](SECURITY.md) for information on our threat model, key handling, and vulnerability reporting process.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
