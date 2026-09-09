# LlamaIndex Integration Guide

Sievrant provides native LlamaIndex integration through two classes:

- **`SievrantRetriever`** implements `BaseRetriever`, returning `NodeWithScore` objects that plug into any LlamaIndex pipeline.
- **`SievrantQueryEngine`** implements `BaseQueryEngine`, wrapping the full Sievrant pipeline (retrieval + conformal gating + optional LLM generation) behind LlamaIndex's standard query interface.

Both classes are drop-in replacements. Your existing LlamaIndex application code does not change.

---

## Why Use Sievrant Instead of the Default LlamaIndex Retriever?

| Feature | LlamaIndex Default | Sievrant + LlamaIndex |
|---|---|---|
| Retrieval | Single-modality (Dense or BM25, not both) | 4-way hybrid (Dense + BM25 + Graph + Tree) |
| Deduplication | None (Top-K raw results) | Submodular greedy selection (mathematically optimal) |
| Coverage guarantee | None | Conformal Prediction gate (e.g. 95% statistical bound) |
| Abstention | Never abstains — always returns an answer | Explicit `ABSTAINED` decision when evidence is insufficient |
| Audit trail | None | Tamper-evident `AuditBlock` on every response (EU AI Act) |
| Multi-tenant ACL | Not built-in | Per-query ACL group filtering |
| Latency | Python overhead | Rust backend, sub-50ms median |

---

## Installation

```bash
pip install "sievrant[llama-index]"
```

This installs `sievrant` plus `llama-index-core`. You can install additional LlamaIndex packages separately:

```bash
pip install llama-index-llms-openai       # OpenAI LLM
pip install llama-index-embeddings-openai  # OpenAI embeddings
```

---

## Quickstart: Retriever

Use `SievrantRetriever` when you want Sievrant to handle retrieval and pass chunks to a LlamaIndex pipeline for post-processing (synthesis, reranking at the Python layer, etc.).

```python
from sievrant.integrations.llama_index import SievrantRetriever

retriever = SievrantRetriever(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="legal-contracts",
    coverage=0.95,         # Conformal coverage guarantee
    token_budget=4000,     # Max tokens across all retrieved chunks
)

nodes = retriever.retrieve("What are the indemnification obligations?")

for node in nodes:
    print(f"Score:   {node.score:.4f}")
    print(f"Text:    {node.text[:200]}")
    print(f"Source:  {node.metadata['source_doc']}")
    print(f"Page:    {node.metadata['page']}")
    print()
```

Each returned `NodeWithScore` carries the full Sievrant metadata on the `node.metadata` dict:

| Key | Type | Description |
|---|---|---|
| `chunk_id` | str | Unique ID of this evidence chunk |
| `source_doc` | str | Original document filename |
| `doc_version_hash` | str | SHA-256 of the source document at ingestion time |
| `path` | str | Hierarchical path within the document (e.g. "Section 3 > Clause 3.2") |
| `page` | int or None | Page number if available |
| `retriever` | str | Which index produced this chunk (VECTOR, BM25, GRAPH, TREE) |
| `rrf_score` | float | Reciprocal Rank Fusion score |
| `rerank_score` | float or None | Cross-encoder rerank score |
| `nonconformity_score` | float | Conformal prediction nonconformity score |
| `submodular_marginal_gain` | float | Marginal coverage gain used in selection |
| `acl_groups` | list[str] | ACL groups this chunk belongs to |
| `freshness_hours` | float or None | Hours since ingestion |
| `token_count` | int | Token count of this chunk |

---

## Quickstart: Query Engine

Use `SievrantQueryEngine` when you want Sievrant to handle the entire pipeline including answer generation and the conformal abstention check. This gives you the full end-to-end Sievrant experience within LlamaIndex.

```python
from sievrant.integrations.llama_index import SievrantQueryEngine

engine = SievrantQueryEngine(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="legal-contracts",
    coverage=0.95,
)

response = engine.query("What are the payment terms?")

print(f"Answer:   {response.response}")
print(f"Decision: {response.metadata['decision']}")        # ANSWERED or ABSTAINED
print(f"Coverage: {response.metadata['coverage_guarantee']:.2%}")
print(f"Latency:  {response.metadata['latency_ms']}ms")
print(f"Audit:    {response.metadata['session_hash']}")

print("\nSource nodes:")
for node in response.source_nodes:
    print(f"  - {node.metadata['source_doc']} (score: {node.score:.4f})")
```

### Handling Abstention

When Sievrant cannot find sufficient evidence to meet the coverage guarantee, `decision` will be `ABSTAINED` and `response.response` will be a clear message explaining the situation. Always check this in production:

```python
if response.metadata["decision"] == "ABSTAINED":
    # Route to a human agent, escalate, or return a safe fallback
    return "I don't have enough information to answer that question confidently."
```

---

## Drop-In for Existing LlamaIndex Pipelines

### Replacing the Default Retriever in a `RetrieverQueryEngine`

```python
from sievrant.integrations.llama_index import SievrantRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

retriever = SievrantRetriever(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="my-corpus",
)

llm = OpenAI(model="gpt-4o-mini")
query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

response = query_engine.query("What is the contract value?")
print(response)
```

### Using with a `RouterQueryEngine`

You can include `SievrantQueryEngine` as one of the tools in a LlamaIndex `RouterQueryEngine`:

```python
from sievrant.integrations.llama_index import SievrantQueryEngine
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

legal_engine = SievrantQueryEngine(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="legal-contracts",
)

finance_engine = SievrantQueryEngine(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="financial-reports",
)

router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        QueryEngineTool.from_defaults(
            legal_engine,
            name="legal",
            description="Use for questions about contracts, terms, and legal obligations",
        ),
        QueryEngineTool.from_defaults(
            finance_engine,
            name="finance",
            description="Use for questions about revenue, expenses, and financial data",
        ),
    ],
)

response = router.query("What was the Q3 revenue?")
```

### Adding to a LlamaIndex SubQuestionQueryEngine

```python
from sievrant.integrations.llama_index import SievrantQueryEngine
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

sievrant_engine = SievrantQueryEngine(
    base_url="http://localhost:8080",
    api_key="sk-your-key",
    corpus="annual-reports",
)

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool.from_defaults(
            sievrant_engine,
            name="annual_reports",
            description="Contains all annual reports from 2019 to 2024",
        )
    ],
    llm=OpenAI(model="gpt-4o"),
    verbose=True,
)

# This will decompose the question into sub-questions automatically
response = sub_question_engine.query(
    "Compare the revenue growth between 2022 and 2023 and explain the key drivers."
)
```

---

## Async Usage

Both `SievrantRetriever` and `SievrantQueryEngine` support async natively. Use `_aretrieve` and `_aquery` directly in async contexts:

```python
import asyncio
from sievrant.integrations.llama_index import SievrantRetriever
from llama_index.core.schema import QueryBundle

async def main():
    retriever = SievrantRetriever(
        base_url="http://localhost:8080",
        api_key="sk-your-key",
        corpus="my-corpus",
    )
    nodes = await retriever._aretrieve(QueryBundle("What is the risk factor?"))
    for node in nodes:
        print(node.text[:100])

asyncio.run(main())
```

---

## Ingesting Documents

Before querying, ingest your documents using the `SievrantClient` directly:

```python
import asyncio
from sievrant import SievrantClient

async def ingest():
    async with SievrantClient(
        base_url="http://localhost:8080",
        api_key="sk-your-key",
    ) as client:
        result = await client.ingest(
            corpus="legal-contracts",
            documents=[
                {
                    "id": "contract-001",
                    "content": open("contract.txt").read(),
                    "metadata": {
                        "filename": "contract.txt",
                        "department": "legal",
                        "date": "2025-01-01",
                    },
                }
            ],
        )
        print(f"Ingested: {result.ingested}, Failed: {result.failed}")

asyncio.run(ingest())
```

---

## Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_url` | str | required | URL of the Sievrant server |
| `api_key` | str | required | API key for authentication |
| `corpus` | str | required | Corpus to query |
| `coverage` | float | `0.95` | Conformal coverage target in (0, 1) |
| `token_budget` | int | `4000` | Maximum tokens across all evidence chunks |
| `acl_groups` | list[str] | `None` | ACL groups to restrict chunks to |
| `callback_manager` | CallbackManager | `None` | LlamaIndex callback manager for tracing |

---

## Next Steps

- [Python SDK Guide](python-sdk.md) — Full `SievrantClient` API reference
- [Architecture deep dive](../concepts/architecture.md) — How the 10-stage pipeline works
- [Conformal Prediction explained](../concepts/conformal-prediction.md) — The math behind coverage guarantees
- [LangChain Integration](langchain-integration.md) — If you use LangChain instead
