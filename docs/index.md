# Sievrant Documentation

Welcome to the Sievrant documentation. Sievrant is the **statistically audited retrieval decision engine** for enterprise AI — a high-performance Rust microservice that sits between your vector database and your AI agents, providing mathematically guaranteed evidence retrieval.

## Quick Navigation

### Getting Started
- [Quickstart Guide](quickstart.md) — Get running in 5 minutes with Docker or Python
- [Installation](guides/installation.md) — All installation methods in detail

### Core Concepts
- [Architecture](concepts/architecture.md) — How the 10-stage pipeline works
- [Evidence Packet](concepts/evidence-packet.md) — The structured output format
- [Conformal Prediction](concepts/conformal-prediction.md) — The math behind the guarantees
- [Submodular Selection](concepts/submodular-selection.md) — Token budget optimization
- [RRF Fusion](concepts/rrf-fusion.md) — How 4 indexes are fused
- [ACL Security](concepts/acl-security.md) — Multi-tenant access control
- [Calibration](concepts/calibration.md) — Calibrating your coverage target
- [Pipeline Routes](concepts/routes.md) — Fast, Standard, Slow, Cached

### Integration Guides
- [Python SDK](guides/python-sdk.md)
- [Node.js SDK](guides/nodejs-sdk.md)
- [Rust SDK](guides/rust-sdk.md)
- [Docker Compose Deployment](guides/docker-compose.md)
- [LangChain Integration](guides/langchain-integration.md)
- [LlamaIndex Integration](guides/llama-index-integration.md)
- [DSPy Integration](guides/dspy-integration.md)
- [MCP (Claude/Cursor)](guides/mcp-integration.md)
- [Corpus Management](guides/corpus-management.md)
- [ACL Configuration](guides/acl-configuration.md)
- [Observability](guides/observability.md)

### API Reference
- [REST API](api-reference/rest-api.md)
- [gRPC API](api-reference/grpc-api.md)
- [Python SDK Reference](api-reference/python-api.md)
- [Node.js SDK Reference](api-reference/nodejs-api.md)

### Comparison
- [vs LangChain](comparison/vs-langchain.md)
- [vs LlamaIndex](comparison/vs-llamaindex.md)
- [vs Haystack](comparison/vs-haystack.md)
- [Benchmarks](comparison/benchmarks.md)

---

## What is Sievrant?

Sievrant is not a vector database. It is not an agent framework. It occupies a distinct layer — the **calibrated retrieval decision layer** — that currently exists only as scattered academic papers and that nobody has shipped as a usable product.

### The layer diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    Your AI Application                           │
│    (LangChain chain, LlamaIndex agent, DSPy program, custom)    │
└────────────────────────────┬─────────────────────────────────────┘
                             │  Uses Sievrant SDK or REST/gRPC
┌────────────────────────────▼─────────────────────────────────────┐
│                  Sievrant Decision Engine                        │
│   ┌─────────┐  ┌────────┐  ┌───────────┐  ┌──────────────────┐ │
│   │  Query  │  │  RRF   │  │Submodular │  │   Conformal      │ │
│   │ Process │─▶│ Fusion │─▶│ Selection │─▶│   Prediction     │ │
│   └─────────┘  └────────┘  └───────────┘  └──────────────────┘ │
└───────────────────────────┬──────────────────────────────────────┘
                            │  Searches all indexes
┌───────────────────────────▼──────────────────────────────────────┐
│                     Index Layer                                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│   │  Vector  │  │  BM25    │  │  Tree    │  │   Graph      │  │
│   │  (Qdrant)│  │(Tantivy) │  │(Hier.)   │  │  (PPR)       │  │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Why this matters

Standard RAG retrieves top-k chunks by similarity score and trusts the LLM to decide what is relevant. This fails in enterprise settings where:
- Hallucinations carry legal liability
- Regulators require auditability (EU AI Act Articles 12 and 13)
- Multi-tenant deployments require strict data isolation
- Token costs must be tightly managed

Sievrant replaces LLM intuition with mathematics at every decision point.
