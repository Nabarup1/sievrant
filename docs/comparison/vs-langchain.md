# Sievrant vs LangChain

An honest, technical comparison for developers choosing a retrieval layer for enterprise AI applications.

---

## Summary

| Dimension | Sievrant | LangChain |
|---|---|---|
| **What it is** | Retrieval decision engine (Rust microservice) | Agent framework (Python library) |
| **Primary use** | Guaranteed, auditable retrieval with statistical bounds | Orchestrating LLM chains, agents, tools |
| **Should they compete?** | No — Sievrant and LangChain are complementary |

Sievrant and LangChain solve different problems. They are not substitutes. The honest answer is: **use both**. Use Sievrant as your retrieval backend (via `SievrantRetriever`) and LangChain for your agent logic, memory, and tool routing.

---

## When You Need Sievrant (Not Just LangChain)

### Problem 1: Hallucination control

**LangChain with Chroma/Pinecone**:
```
User query → top-5 by cosine similarity → stuff into LLM prompt → answer
```
If none of the top-5 contain the true answer, the LLM invents something. There is no mechanism to detect this.

**Sievrant**:
```
User query → 4-way hybrid search → RRF → rerank → submodular selection →
conformal gate (P ≥ 0.95 the answer is in there) → answer, OR abstain honestly
```
If Sievrant cannot retrieve evidence it is statistically confident in, it returns `ABSTAINED`. No hallucination. The guarantee is mathematical.

---

### Problem 2: Redundant context

**LangChain top-k**:
```
Query: "What are the payment terms?"
Chunk 1: "Monthly rent is $4,500 due on the 1st" (0.87 similarity)
Chunk 2: "Rent of four thousand five hundred dollars is due monthly" (0.85)
Chunk 3: "Payment of $4,500 monthly on the first" (0.83)
Chunk 4: "Late payment incurs 5% fee" (0.79)
Chunk 5: "Security deposit is 2 months' rent" (0.77)
```
Three of five chunks say the same thing. You've burned tokens and crowded out relevant non-redundant content.

**Sievrant submodular selection**:
```
Chunk 1 selected (marginal gain: 0.82) — covers rent amount
Chunk 4 selected (marginal gain: 0.71) — covers late fees (new information)
Chunk 5 selected (marginal gain: 0.58) — covers deposit (new information)
Chunks 2, 3 rejected (marginal gain: 0.04, 0.02) — redundant
```
The context is maximally informative within the token budget. This is the submodular guarantee: (1 - 1/e) ≈ 63% of optimal coverage.

---

### Problem 3: Multi-index retrieval

**LangChain**: Typically calls one vector store. Hybrid search requires manual setup of multiple retrievers and ensemble logic.

**Sievrant**: Four indexes run in parallel and are fused via RRF in every call:
- Vector (dense semantic)
- BM25 (sparse lexical)
- Tree (hierarchical structure)
- Graph (relational multi-hop)

The fusion is automatic, calibration-free, and deterministic.

---

### Problem 4: Audit trails

**LangChain**: No built-in audit mechanism. You build it yourself.

**Sievrant**: Every `EvidencePacket` includes:
- A `session_hash` — SHA-256 of (query, evidence, answer, model, timestamp) for tamper-evidence
- Per-stage latency breakdown (13 stages)
- Which index contributed each chunk
- Conformal non-conformity score per chunk
- Calibration set ID used
- `empirical_coverage_last_30d` for SLO tracking

This satisfies EU AI Act Articles 12 and 13 record-keeping requirements out of the box.

---

### Problem 5: Multi-tenant ACL

**LangChain**: Namespace support exists in some vector stores but requires manual implementation at the application layer.

**Sievrant**: ACL groups are enforced at the pre-retrieval filtering stage. The ACL check runs before any chunk is fetched from any index, so there is no possibility of an unauthorized chunk leaking into a result set.

---

## What LangChain Does That Sievrant Does Not

Sievrant is not:
- An agent framework
- A tool-calling orchestrator
- A memory system
- A prompt template engine
- An LLM routing layer

LangChain excels at all of these. The designed architecture for serious enterprise deployments is:

```
LangChain Agent
  ├── Memory (LangChain ConversationBufferMemory)
  ├── Tools (LangChain tool calls)
  │    ├── SievrantRetriever ← Sievrant handles retrieval
  │    ├── Calculator
  │    └── Web search
  └── LLM (any provider)
```

---

## Migration from LangChain Retrieval

If you currently use a LangChain vector store retriever, switching to Sievrant is two lines:

```python
# Before
from langchain_community.vectorstores import Chroma
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# After
from sievrant.integrations.langchain import SievrantRetriever
retriever = SievrantRetriever(base_url="...", api_key="...", corpus="...")
```

Everything else in your chain — `RetrievalQA`, `ConversationalRetrievalChain`, LCEL pipelines, memory — requires zero changes.
