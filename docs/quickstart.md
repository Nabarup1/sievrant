# Quickstart

Get Sievrant running and serving your first query in under 5 minutes.

## Prerequisites

Choose one of the following setups based on how you want to run Sievrant:

- **Docker** (recommended for evaluation): Docker 24+ and Docker Compose v2
- **Python client only**: Python 3.10+ and a running Sievrant server
- **Rust embed**: Rust 1.78+ and Cargo

---

## Option 1: Docker (Recommended)

This is the fastest way to get a fully functional Sievrant instance running, including the Qdrant vector database.

### Step 1: Clone and configure

```bash
git clone https://github.com/Nabarup1/sievrant
cd sievrant
cp .env.example .env
```

Open `.env` and fill in your LLM provider key:

```bash
# Minimum required
OPENAI_API_KEY=sk-your-openai-key

# Optional: use Claude instead
# ANTHROPIC_API_KEY=sk-ant-your-key

# The API key clients will use to authenticate to Sievrant
SIEVRANT_API_KEYS=sk-demo-key:tenant-demo
```

### Step 2: Start everything

```bash
docker-compose up -d
```

This starts:
- `sievrant-server` on port 8080 (REST) and port 50051 (gRPC)
- `qdrant` on port 6333 (vector store)
- `jaeger` on port 16686 (distributed traces, optional)

### Step 3: Verify the server is healthy

```bash
curl http://localhost:8080/v1/health
```

Expected response:

```json
{"status": "ok", "version": "0.1.0"}
```

### Step 4: Ingest your first document

```bash
curl -X POST http://localhost:8080/v1/ingest \
  -H "Authorization: Bearer sk-demo-key" \
  -H "Content-Type: application/json" \
  -d '{
    "corpus": "my-first-corpus",
    "documents": [
      {
        "id": "doc-001",
        "content": "The lease agreement commences on January 1, 2025. The lease term is 60 months, ending December 31, 2029. Monthly rent is $4,500 USD, payable on the first of each month.",
        "metadata": {"filename": "lease.pdf", "author": "Legal Team"}
      }
    ]
  }'
```

### Step 5: Query with a coverage guarantee

```bash
curl -X POST http://localhost:8080/v1/query \
  -H "Authorization: Bearer sk-demo-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the monthly rent amount?",
    "corpus": "my-first-corpus",
    "coverage": 0.95
  }'
```

Expected response:

```json
{
  "query_id": "01932a7b-...",
  "decision": "ANSWERED",
  "answer": "The monthly rent is $4,500 USD, payable on the first of each month.",
  "evidence_count": 1,
  "coverage_guarantee": 0.95,
  "latency_ms": 43,
  "cache_hit": false,
  "route": "FAST"
}
```

The `decision: ANSWERED` means Sievrant's conformal gate confirmed, with at least 95% statistical confidence, that the retrieved evidence contains the true answer.

---

## Option 2: Python SDK

Install the SDK:

```bash
pip install sievrant
```

Then run this script against a running server:

```python
import asyncio
from sievrant import SievrantClient

async def main():
    client = SievrantClient(
        base_url="http://localhost:8080",
        api_key="sk-demo-key"
    )

    # Check server health
    health = await client.health()
    print(f"Server status: {health.status}")

    # Ingest a document
    await client.ingest(
        corpus="my-first-corpus",
        documents=[{
            "id": "doc-001",
            "content": "The lease term is 60 months, ending December 31, 2029. Monthly rent is $4,500 USD.",
            "metadata": {"filename": "lease.pdf"}
        }]
    )

    # Query
    result = await client.query(
        corpus="my-first-corpus",
        query="What is the monthly rent?",
        coverage=0.95
    )

    print(f"Decision: {result.decision}")
    print(f"Answer: {result.answer}")
    print(f"Evidence chunks: {len(result.evidence)}")
    print(f"Latency: {result.latency_ms}ms")

    for chunk in result.evidence:
        print(f"  - [{chunk.retriever}] score={chunk.rrf_score:.4f}: {chunk.text[:80]}...")

asyncio.run(main())
```

---

## Option 3: Node.js / TypeScript

```bash
npm install sievrant
```

```typescript
import { SievrantClient } from 'sievrant';

const client = new SievrantClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'sk-demo-key',
});

// Ingest
await client.ingest({
  corpus: 'my-first-corpus',
  documents: [{
    id: 'doc-001',
    content: 'The lease term is 60 months. Monthly rent is $4,500.',
  }],
});

// Query
const result = await client.query({
  corpus: 'my-first-corpus',
  query: 'What is the monthly rent?',
  coverage: 0.95,
});

console.log(result.decision);   // "ANSWERED"
console.log(result.answer);     // "The monthly rent is $4,500."
```

---

## Option 4: Rust (native embed)

Add to `Cargo.toml`:

```toml
[dependencies]
sievrant = "0.1"
tokio = { version = "1", features = ["full"] }
```

```rust
use sievrant::{Corpus, CorpusConfig, Document, RetrievalConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let corpus = Corpus::new("my-first-corpus", CorpusConfig::default()).await?;

    corpus.ingest(vec![Document {
        id: "doc-001".into(),
        content: "The lease term is 60 months. Monthly rent is $4,500.".into(),
        ..Default::default()
    }]).await?;

    let packet = corpus.retrieve(
        "What is the monthly rent?",
        &"my-first-corpus",
        RetrievalConfig { coverage: 0.95, ..Default::default() }
    ).await?;

    match packet.decision {
        Decision::Answered => println!("Answer: {:?}", packet.answer),
        Decision::Abstained => println!("Sievrant abstained: insufficient evidence"),
        Decision::EvidenceOnly => println!("Evidence: {:?}", packet.evidence),
    }

    Ok(())
}
```

---

## Next Steps

- Read [Architecture](concepts/architecture.md) to understand the 10-stage pipeline
- Set up [LangChain integration](guides/langchain-integration.md) to use Sievrant as a drop-in retriever
- Configure [ACL rules](guides/acl-configuration.md) for multi-tenant document access
- Run [calibration](guides/calibration-guide.md) on your corpus to tune the conformal threshold
- Explore the [REST API reference](api-reference/rest-api.md) for all available endpoints
