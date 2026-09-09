# Python SDK Guide

The Sievrant Python SDK is a fully-typed async HTTP client for the Sievrant server. It wraps the REST API in a Pythonic interface and provides first-class integrations with LangChain, LlamaIndex, and DSPy.

## Installation

```bash
pip install sievrant
```

Python 3.10+ required.

**Optional extras** for framework integrations:

```bash
pip install "sievrant[langchain]"      # LangChain integration
pip install "sievrant[llama-index]"    # LlamaIndex integration
pip install "sievrant[dspy]"           # DSPy integration
pip install "sievrant[all]"            # All integrations
```

---

## Basic Usage

### Async client (recommended)

```python
import asyncio
from sievrant import SievrantClient

async def main():
    async with SievrantClient(
        base_url="http://localhost:8080",
        api_key="sk-your-key"
    ) as client:
        
        # Health check
        health = await client.health()
        print(f"Sievrant {health.version} — {health.status}")
        
        # Ingest documents
        await client.ingest(
            corpus="legal-docs",
            documents=[
                {
                    "id": "contract-001",
                    "content": "The lease commences January 1, 2025 for 60 months.",
                    "metadata": {"filename": "lease.pdf", "author": "Legal Dept"}
                }
            ]
        )
        
        # Query
        result = await client.query(
            corpus="legal-docs",
            query="When does the lease commence?",
            coverage=0.95,
        )
        
        print(f"Decision:  {result.decision}")
        print(f"Answer:    {result.answer}")
        print(f"Latency:   {result.latency_ms}ms")
        print(f"Evidence:  {len(result.evidence)} chunks")

asyncio.run(main())
```

### Sync wrapper (for notebooks and simple scripts)

```python
from sievrant import SievrantClient

client = SievrantClient.sync(
    base_url="http://localhost:8080",
    api_key="sk-your-key"
)

result = client.query_sync(corpus="legal-docs", query="When does the lease start?")
print(result.decision)
```

---

## Client Configuration

```python
from sievrant import SievrantClient

client = SievrantClient(
    base_url="http://localhost:8080",   # Sievrant server URL
    api_key="sk-your-key",             # Bearer token
    timeout=30.0,                       # Request timeout in seconds (default 30)
    max_retries=3,                      # Retry on 5xx (default 3)
    retry_backoff=1.5,                  # Exponential backoff multiplier
)
```

---

## `query()` Reference

```python
result = await client.query(
    corpus="legal-docs",           # Required: corpus name
    query="What is the rent?",     # Required: natural language query
    coverage=0.95,                 # Coverage target (default 0.95)
    token_budget=4000,             # Max tokens across evidence (default 4000)
    acl_groups=["legal", "hr"],    # ACL groups for this user (default: caller's groups)
    skip_cache=False,              # Skip semantic cache (default False)
    route="AUTO",                  # AUTO, FAST, STANDARD, SLOW (default AUTO)
)
```

**Returns**: `EvidencePacket`

---

## `ingest()` Reference

```python
await client.ingest(
    corpus="legal-docs",           # Corpus to ingest into (created automatically if needed)
    documents=[
        {
            "id": "doc-001",                     # Required: unique document ID
            "content": "Full text content...",   # Required: document text
            "metadata": {                         # Optional: arbitrary metadata
                "filename": "lease.pdf",
                "author": "Legal Team",
                "created_at": "2025-01-01T00:00:00Z",
            },
            "acl_groups": ["legal", "finance"],  # Optional: access control groups
        }
    ]
)
```

---

## `EvidencePacket` Model

```python
from sievrant.models import EvidencePacket, EvidenceChunk, Decision

# After a query
result: EvidencePacket = await client.query(...)

# Top-level fields
result.query_id: str          # UUID for this request
result.decision: Decision     # Decision.ANSWERED, Decision.ABSTAINED, Decision.EVIDENCE_ONLY
result.answer: str | None     # Generated answer or None
result.coverage_guarantee: float
result.empirical_coverage_last_30d: float | None
result.route: str
result.cache_hit: bool
result.latency_ms: int
result.cost_usd: float | None

# Evidence chunks
for chunk in result.evidence:
    chunk.chunk_id: str
    chunk.source_doc: str
    chunk.doc_version_hash: str
    chunk.path: list[str]
    chunk.page: int | None
    chunk.text: str
    chunk.retriever: str           # "vector_index", "bm25_index", etc.
    chunk.rrf_score: float
    chunk.rerank_score: float | None
    chunk.nonconformity_score: float
    chunk.submodular_marginal_gain: float
    chunk.acl_groups: list[str]
    chunk.token_count: int

# Audit block
result.audit.model_version: str | None
result.audit.session_hash: str       # SHA-256 tamper-evident hash
result.audit.calibration_set_id: str
result.audit.retrieval_stages_ms     # Per-stage latency breakdown
```

---

## LangChain Integration

```python
from sievrant.integrations.langchain import SievrantRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Drop into any LangChain chain
retriever = SievrantRetriever(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="legal-docs",
    coverage=0.95,
    k=5,                  # Max number of chunks to return as LangChain Documents
)

llm = ChatOpenAI(model="gpt-4o")
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

answer = chain.invoke("What is the monthly rent?")
print(answer["result"])
```

`SievrantRetriever` implements `BaseRetriever` and converts `EvidenceChunk` objects to LangChain `Document` objects with full metadata preserved in `Document.metadata`.

### Handling abstentions in LangChain

```python
from sievrant.integrations.langchain import SievrantRetriever, SievrantAbstainedError

retriever = SievrantRetriever(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="legal-docs",
    raise_on_abstain=True,   # Raise SievrantAbstainedError instead of returning empty list
)

try:
    docs = await retriever.aget_relevant_documents("ambiguous question")
except SievrantAbstainedError:
    # Handle gracefully: log, escalate, show fallback UI
    pass
```

---

## LlamaIndex Integration

```python
from sievrant.integrations.llama_index import SievrantQueryEngine
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-4o")

engine = SievrantQueryEngine(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="legal-docs",
    coverage=0.95,
)

response = engine.query("What are the payment terms?")
print(response.response)
print(response.source_nodes)
```

---

## DSPy Integration

```python
import dspy
from sievrant.integrations.dspy import SievrantRM

# Use Sievrant as a DSPy retrieval module
dspy.settings.configure(
    rm=SievrantRM(
        base_url="http://localhost:8080",
        api_key="sk-your-key",
        corpus="legal-docs",
    )
)

class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

rag = RAG()
answer = rag("What is the lease term?")
print(answer.answer)
```

---

## Error Handling

```python
from sievrant.exceptions import (
    SievrantError,          # Base exception
    AuthError,              # 401: Invalid or missing API key
    CorpusNotFoundError,    # 404: Corpus does not exist
    SievrantServerError,    # 5xx: Server-side error
    SievrantTimeoutError,   # Request timed out
    SievrantAbstainedError, # Decision was ABSTAINED (optional, see raise_on_abstain)
)

try:
    result = await client.query(corpus="missing-corpus", query="test")
except CorpusNotFoundError as e:
    print(f"Corpus not found: {e.corpus}")
except AuthError:
    print("Invalid API key")
except SievrantTimeoutError:
    print("Request timed out after 30s")
except SievrantError as e:
    print(f"Unexpected error: {e}")
```

---

## Environment Variables

The client reads these environment variables automatically if constructor arguments are not provided:

```bash
SIEVRANT_BASE_URL=http://localhost:8080
SIEVRANT_API_KEY=sk-your-key
SIEVRANT_TIMEOUT=30
```

```python
# These are equivalent:
client = SievrantClient(base_url="http://localhost:8080", api_key="sk-key")
client = SievrantClient()  # Reads from environment
```

---

## Logging

The SDK uses Python's standard `logging` module under the `sievrant` logger:

```python
import logging
logging.getLogger("sievrant").setLevel(logging.DEBUG)
```

---

## Testing Against a Local Server

For unit tests without a running server, use the built-in `MockSievrantClient`:

```python
from sievrant.testing import MockSievrantClient
from sievrant.models import Decision

async def test_my_agent():
    mock = MockSievrantClient()
    mock.mock_query(
        corpus="test",
        decision=Decision.ANSWERED,
        answer="The lease is 60 months.",
        evidence=[...],
    )
    
    # Inject mock into your code
    result = await mock.query(corpus="test", query="What is the lease term?")
    assert result.decision == Decision.ANSWERED
```
