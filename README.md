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

Sievrant hybrid retrieval (Vector + BM25 + RRF) outperforms single-modality retrieval across BEIR datasets. Full benchmark methodology and results in [docs/comparison/benchmarks.md](docs/comparison/benchmarks.md).

| Benchmark | Sievrant Hybrid | BM25 only | Dense only |
|---|---|---|---|
| MS MARCO (MRR@10) | 0.38 | 0.18 | 0.33 |
| NQ (Recall@100) | 0.87 | 0.74 | 0.81 |
| HotpotQA (Recall@100) | 0.79 | 0.63 | 0.71 |
| Latency p50 (ms) | 45 | 12 | 38 |
| Latency p99 (ms) | 180 | 55 | 165 |

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

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
