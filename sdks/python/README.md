# Sievrant Python SDK

[![PyPI version](https://img.shields.io/pypi/v/sievrant.svg)](https://pypi.org/project/sievrant/)
[![Python versions](https://img.shields.io/pypi/pyversions/sievrant.svg)](https://pypi.org/project/sievrant/)
[![License](https://img.shields.io/github/license/Nabarup1/sievrant.svg)](https://github.com/Nabarup1/sievrant/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Nabarup1/sievrant.svg)](https://github.com/Nabarup1/sievrant/stargazers)

**Statistically audited retrieval for enterprise AI.**

Sievrant is a retrieval decision engine that guarantees the correctness of your Retrieval-Augmented Generation (RAG) pipelines using Conformal Prediction.

## The Problem

Traditional RAG pipelines suffer from unpredictable failure modes:
*   **Silent Hallucinations:** When retrieval fails to find the answer, the LLM hallucinates, and the system fails silently.
*   **Uncalibrated Confidence:** It is impossible to know when to trust the LLM's answer and when to escalate to a human.
*   **Wasted Compute:** Rerankers and LLMs are expensive. Running them on every query, even simple ones, wastes resources.
*   **Compliance Risks:** In enterprise settings, lack of statistical guarantees prevents AI from being deployed in high-stakes environments.

## The Solution

Sievrant replaces your standard retriever with a **Retrieval Decision Engine**:
*   **Mathematical Guarantees:** Uses Conformal Prediction to provide a strict lower bound (e.g., 95%) on the probability that the retrieved context contains the true answer.
*   **Abstention:** Automatically decides when to answer and when to `ABSTAIN` (escalate to a human) based on your specified coverage guarantee.
*   **Dynamic Routing:** Routes simple queries to fast, cheap pipelines (e.g., Cache -> BM25) and complex queries to slower, expensive pipelines (e.g., Vector -> GraphRAG -> Reranker) based on real-time difficulty estimation.
*   **Auditability:** Every decision is cryptographically hashed and logged with full provenance, latency breakdowns, and compliance metadata.

## Installation

Install the base package via `pip`:

```bash
pip install sievrant
```

### Optional Dependencies

Sievrant provides optional dependencies for framework integrations:

*   **LangChain**: `pip install "sievrant[langchain]"`
*   **LlamaIndex**: `pip install "sievrant[llama-index]"`
*   **DSPy**: `pip install "sievrant[dspy]"`
*   **All**: `pip install "sievrant[all]"`

## Quickstart

Sievrant provides both asynchronous and synchronous clients. The asynchronous client is recommended for production applications to maximize throughput.

### Async Client (Recommended)

```python
import asyncio
import os
from sievrant import AsyncSievrantClient

async def main():
    # Initializes using SIEVRANT_BASE_URL and SIEVRANT_API_KEY environment variables
    client = AsyncSievrantClient()
    
    # 1. Ingest some documents
    await client.ingest(
        corpus="legal-contracts",
        documents=[
            {"id": "doc1", "content": "The lease commences on January 1, 2025."},
            {"id": "doc2", "content": "Rent is $2,000 per month, due on the 1st."}
        ]
    )
    
    # 2. Query with a 95% coverage guarantee
    response = await client.query(
        corpus="legal-contracts",
        query="When does the lease commence?",
        coverage=0.95
    )
    
    # 3. Handle the decision
    if response.decision == "ANSWERED":
        print(f"Answer: {response.answer}")
        print(f"Evidence used: {response.evidence[0].text}")
    elif response.decision == "ABSTAINED":
        print("Sievrant abstained: Escalating to human agent.")
    elif response.decision == "EVIDENCE_ONLY":
        print("Evidence returned without generation.")

if __name__ == "__main__":
    asyncio.run(main())
```

### Sync Client

```python
import os
from sievrant import SievrantClient

def main():
    client = SievrantClient(
        base_url=os.getenv("SIEVRANT_BASE_URL", "http://localhost:8080"),
        api_key=os.getenv("SIEVRANT_API_KEY", "your-api-key")
    )
    
    response = client.query(
        corpus="legal-contracts",
        query="What is the monthly rent?",
        coverage=0.99
    )
    
    print(f"Decision: {response.decision}")
    if response.answer:
        print(f"Answer: {response.answer}")

if __name__ == "__main__":
    main()
```

## Reading the Response (`EvidencePacket`)

The `EvidencePacket` returned by the `query` method contains everything you need to understand *why* Sievrant made a specific decision.

```python
response = client.query(corpus="knowledge-base", query="How do I reset my password?")

# The decision can be 'ANSWERED', 'ABSTAINED', or 'EVIDENCE_ONLY'
print(response.decision)

# The generated answer (if ANSWERED)
print(response.answer)

# The list of EvidenceChunk objects that passed the conformal gate
for chunk in response.evidence:
    print(chunk.chunk_id)
    print(chunk.source_doc)
    print(chunk.text)
    print(chunk.retriever) # e.g., 'VECTOR_INDEX' or 'BM25'
    print(chunk.nonconformity_score) # Lower is more trustworthy

# The nominal coverage guarantee requested
print(response.coverage_guarantee)

# Total latency of the request in milliseconds
print(response.latency_ms)

# Which pipeline path was taken (e.g., 'FAST', 'STANDARD', 'SLOW')
print(response.route)

# Was this served from the semantic cache?
print(response.cache_hit)

# The AuditBlock containing compliance metadata
print(response.audit.session_hash)
print(response.audit.retrieval_stages_ms) # Latency breakdown per stage
```

### Handling `ABSTAINED`

When Sievrant cannot guarantee that the retrieved evidence contains the answer at the requested confidence level (e.g., 95%), it will abstain.

```python
response = client.query(corpus="medical-records", query="What is the patient's diagnosis?", coverage=0.99)

if response.decision == "ABSTAINED":
    # The system could not reach 99% confidence.
    # We gracefully fallback to a human expert.
    escalate_ticket_to_human("What is the patient's diagnosis?", patient_id="123")
```

## Ingestion

Sievrant automatically creates a corpus if it does not exist when you first ingest documents into it.

### Basic Ingestion

```python
client.ingest(
    corpus="employee-handbook",
    documents=[
        {"id": "doc1", "content": "The company observes 10 paid holidays."},
        {"id": "doc2", "content": "Employees get 15 days of PTO."}
    ]
)
```

### Ingestion from Files

Use the helper method to easily ingest from local files.

```python
client.ingest_file(
    corpus="financial-reports",
    file_path="./q3_earnings.pdf",
    metadata={"year": 2023, "quarter": "Q3"}
)
```

### Metadata and ACLs

You can attach metadata and Access Control List (ACL) groups to restrict who can retrieve specific documents.

```python
client.ingest(
    corpus="internal-wiki",
    documents=[
        {
            "id": "hr-policies",
            "content": "Salary bands for engineering roles...",
            "metadata": {"department": "HR", "confidential": True},
            "acl_groups": ["hr-team", "executives"]
        }
    ]
)

# When querying, specify the user's ACL groups
response = client.query(
    corpus="internal-wiki",
    query="What is the salary band for L4?",
    acl_groups=["engineering"] # This user will NOT retrieve the hr-policies document
)
```

## Framework Integrations

Sievrant seamlessly integrates with the most popular AI frameworks, allowing you to upgrade your existing RAG pipelines to be statistically audited.

### LangChain

Use Sievrant as a Retriever in LangChain.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sievrant.integrations.langchain import SievrantRetriever

retriever = SievrantRetriever(
    corpus="my-corpus",
    coverage=0.90, # 90% confidence
    decision_mode="EVIDENCE_ONLY" # Let LangChain's LLM handle generation
)

llm = ChatOpenAI(model="gpt-4o")
prompt = PromptTemplate.from_template(
    "Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
)

# Build a standard LangChain RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

result = chain.invoke("What is Sievrant?")
print(result.content)
```

### LlamaIndex

Use Sievrant as a QueryEngine or Retriever in LlamaIndex.

```python
from sievrant.integrations.llama_index import SievrantQueryEngine

# Acts as a full QueryEngine, utilizing Sievrant's internal generation
query_engine = SievrantQueryEngine(
    corpus="my-corpus",
    coverage=0.95
)

response = query_engine.query("What is Sievrant?")
print(response.response)

# Access the full EvidencePacket via metadata
if "sievrant_packet" in response.metadata:
    packet = response.metadata["sievrant_packet"]
    print(f"Decision: {packet.decision}")
```

### DSPy

Use Sievrant as a Retrieval Model (RM) in DSPy.

```python
import dspy
from sievrant.integrations.dspy import SievrantRM

# Configure DSPy to use Sievrant for retrieval
retriever_model = SievrantRM(corpus="my-corpus", coverage=0.95)
dspy.settings.configure(rm=retriever_model)

# Define a DSPy Signature
class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Define a DSPy Module
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(QA)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

uncompiled_rag = RAG()
result = uncompiled_rag(question="What is Sievrant?")
print(result.answer)
```

## Environment Variables

Sievrant clients automatically pick up the following environment variables:

| Variable | Description | Default |
|---|---|---|
| `SIEVRANT_BASE_URL` | The URL of your Sievrant server. | `http://localhost:8080` |
| `SIEVRANT_API_KEY` | Your authentication API key. | *(Required)* |

## Error Handling

Sievrant maps API errors to specific exception classes in `sievrant.exceptions`.

| Exception Class | HTTP Status | Description |
|---|---|---|
| `AuthError` | 401 | The API key is missing or invalid. |
| `CorpusNotFoundError` | 404 | The requested corpus does not exist. |
| `RateLimitError` | 429 | The rate limit has been exceeded. Check `retry_after`. |
| `SievrantServerError` | 500+ | An internal error occurred on the Sievrant server. |
| `SievrantTimeoutError` | N/A | The request timed out. |
| `SievrantError` | N/A | Base class for all Sievrant exceptions. |

### Example

```python
from sievrant import SievrantClient
from sievrant.exceptions import CorpusNotFoundError, AuthError, SievrantTimeoutError

client = SievrantClient()

try:
    response = client.query(corpus="non-existent", query="Hello?")
except CorpusNotFoundError as e:
    print(f"Corpus not found: {e.corpus}")
except AuthError as e:
    print("Invalid API Key!")
except SievrantTimeoutError as e:
    print(f"Request timed out after {e.timeout_ms}ms")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

## Advanced Features

Sievrant includes advanced capabilities for optimizing your RAG pipelines.

### GraphRAG Entity Extraction

Extract structured entities and relationships from text to build Knowledge Graphs for Graph-based retrieval.

```python
from sievrant.advanced import EntityExtractor

extractor = EntityExtractor(client)
entities = extractor.extract(
    text="Satya Nadella is the CEO of Microsoft, headquartered in Redmond.",
    schema={"Person": ["name", "role"], "Company": ["name", "location"]}
)
print(entities)
```

### RAPTOR Tree Summarization

Build hierarchical summaries of large documents using the RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) methodology.

```python
from sievrant.advanced import TreeSummarizer

summarizer = TreeSummarizer(client)
tree = summarizer.build_tree(
    document_text="...", # Very long document
    levels=3 # Create 3 levels of abstraction
)

# Ingest the resulting tree nodes into your corpus
client.ingest(corpus="my-corpus", documents=tree.to_documents())
```

### Prompt Optimization

Automatically optimize your system prompts based on a small dataset of examples.

```python
from sievrant.advanced import PromptOptimizer

optimizer = PromptOptimizer(client)
optimized_prompt = optimizer.optimize(
    initial_prompt="You are a helpful assistant.",
    dataset=[
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Capital of France?", "answer": "Paris"}
    ]
)
print(optimized_prompt)
```

## Full API Reference

For the complete API reference, visit the [Documentation](https://github.com/Nabarup1/sievrant/tree/main/docs).

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/Nabarup1/sievrant/blob/main/CONTRIBUTING.md) for details on how to set up your development environment, run tests, and submit Pull Requests.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/Nabarup1/sievrant/blob/main/LICENSE) file for details.
