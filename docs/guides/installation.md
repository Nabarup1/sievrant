# Installation Guide

This guide covers all installation methods for Sievrant.

---

## Server Installation

The Sievrant server is the binary that runs the 10-stage retrieval pipeline. Choose the method that best fits your workflow.

### Docker Compose (Recommended)

The fastest way to get a complete Sievrant stack running, including Qdrant for vector storage.

```bash
git clone https://github.com/Nabarup1/sievrant
cd sievrant
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=sk-your-key              # LLM provider key
SIEVRANT_API_KEYS=sk-demo:tenant-demo   # API key format: key:tenant_name
SIEVRANT_LOG_LEVEL=info
QDRANT_URL=http://qdrant:6333
```

Start the stack:

```bash
docker-compose up -d
```

Check health:

```bash
curl http://localhost:8080/v1/health
```

### Rust Binary (Cargo)

Requires Rust 1.78+ and the ONNX Runtime for local reranking models.

```bash
git clone https://github.com/Nabarup1/sievrant
cd sievrant
cargo build --release -p sievrant-server
./target/release/sievrant-server --config config.toml
```

### Install via Cargo (coming soon to crates.io)

```bash
cargo install sievrant-server
sievrant-server --config config.toml
```

---

## Python SDK

Requires Python 3.10+.

```bash
pip install sievrant
```

### With LangChain integration

```bash
pip install "sievrant[langchain]"
```

### With LlamaIndex integration

```bash
pip install "sievrant[llama-index]"
```

### With DSPy integration

```bash
pip install "sievrant[dspy]"
```

### All integrations

```bash
pip install "sievrant[all]"
```

---

## Node.js SDK

Requires Node.js 18+.

```bash
npm install sievrant
```

---

## Rust SDK (native embed)

Add to `Cargo.toml`:

```toml
[dependencies]
sievrant = "0.1"
```

This embeds the entire retrieval engine in your Rust binary. No network hop, no server process.

---

## Environment Configuration

### Required environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (or `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`) |
| `SIEVRANT_API_KEYS` | Comma-separated `key:tenant` pairs, e.g. `sk-prod:acme,sk-dev:test` |

### Optional environment variables

| Variable | Default | Description |
|---|---|---|
| `SIEVRANT_HOST` | `0.0.0.0` | Server bind address |
| `SIEVRANT_REST_PORT` | `8080` | REST API port |
| `SIEVRANT_GRPC_PORT` | `50051` | gRPC port |
| `SIEVRANT_LOG_LEVEL` | `info` | Log level: `trace`, `debug`, `info`, `warn`, `error` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector store URL |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | (unset) | OpenTelemetry collector endpoint |
| `SIEVRANT_RERANKER` | `local` | Reranker: `local`, `cohere`, `jina` |
| `COHERE_API_KEY` | (unset) | Required if `SIEVRANT_RERANKER=cohere` |

---

## Verifying the Installation

After starting the server, verify all components are working:

```bash
# Health check
curl http://localhost:8080/v1/health

# Ingest a test document
curl -X POST http://localhost:8080/v1/ingest \
  -H "Authorization: Bearer sk-demo" \
  -H "Content-Type: application/json" \
  -d '{"corpus":"test","documents":[{"id":"test-001","content":"Sievrant is working correctly."}]}'

# Query it back
curl -X POST http://localhost:8080/v1/query \
  -H "Authorization: Bearer sk-demo" \
  -H "Content-Type: application/json" \
  -d '{"corpus":"test","query":"Is Sievrant working?"}'
```

Expected response should have `"decision": "ANSWERED"`.

### Swagger UI

Navigate to `http://localhost:8080/swagger-ui` in your browser to explore the interactive API documentation.
