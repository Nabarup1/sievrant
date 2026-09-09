# Sievrant: Project Memory

This is a living document tracking architectural decisions, open questions, lessons learned, and useful references. Update it as the project evolves.

---

## Architectural Decisions

### AD 001: Rust as primary language
**Date**: August 30, 2026
**Decision**: Build the core engine in Rust, use Python only for benchmarking, evaluation, and calibration data tooling.
**Rationale**: Qdrant (Rust) benchmarks at p50 ~4ms with 2 to 3x less memory than Go alternatives. Microsoft DiskANN team rewrote in Rust for composable, embeddable core. Hiring pool is smaller but dependency strategy (lean on Qdrant, Tantivy, petgraph) limits the amount of Rust code we actually write.

### AD 002: Conformal prediction as first class trust gate
**Date**: August 30, 2026
**Decision**: Ship conformal prediction as an always on, non optional component of the retrieval pipeline.
**Rationale**: No commercial RAG product ships this as of August 2026. Research exists (Conformal RAG, C RAG, URAG, TRAQ) but no products. This is the core technical differentiator.
**Risk**: Exchangeability assumption. Mitigation: per tenant calibration, live coverage monitoring, explicit drift alerts.

### AD 003: Evidence Packet as product surface
**Date**: August 30, 2026
**Decision**: The Evidence Packet JSON schema is the API contract. Every upstream component exists to fill it correctly.
**Rationale**: Agents need structured, machine readable context with full metadata. Text blobs are not sufficient for compound AI systems. The packet also doubles as a compliance artifact (EU AI Act Articles 12/13).

### AD 004: Pre retrieval ACL enforcement
**Date**: August 30, 2026
**Decision**: Access control filters are applied at the index query level, before scoring. No post filtering.
**Rationale**: Post filtering leaks unauthorized data into vector scoring, reranking, and logging. Security researchers in 2026 classify post filtering as a design vulnerability. Embedding inversion attacks can reconstruct source content from vectors.

### AD 005: Adaptive query routing
**Date**: August 30, 2026
**Decision**: Classify queries into FAST, STANDARD, and SLOW paths using a lightweight TF IDF classifier.
**Rationale**: RAGRouter Bench research shows TF IDF features outperform embedding based classifiers for complexity routing. This saves 60 to 80 percent compute on simple queries while preserving accuracy on complex ones.

### AD 006: MCP server as distribution channel
**Date**: August 30, 2026
**Decision**: Expose Sievrant as a Model Context Protocol server in addition to HTTP/gRPC.
**Rationale**: MCP is the 2026 standard for agent to tool communication. Every major agent framework (Claude, Cursor, VS Code, custom) supports MCP. This eliminates custom integration code and dramatically expands the addressable market.

### AD 007: Three tier semantic cache
**Date**: August 30, 2026
**Decision**: Implement exact match KV, then semantic similarity, then full pipeline as three cache tiers.
**Rationale**: Production workloads show 70 to 90 percent reduction in costs and latency with semantic caching. Required for viable per query pricing. Tenant isolation is mandatory to prevent data leakage.

---

## Open Questions

- [ ] What is the minimum viable calibration set size where conformal guarantees actually hold in practice? Literature says 100 but we need to validate empirically across different corpus types.
- [ ] Should the cross encoder reranker default to local ONNX inference or API based (Cohere)? Latency vs simplicity tradeoff needs benchmarking on actual hardware.
- [ ] How should we handle multi language corpora? BM25 tokenization and embedding models may need per language configuration.
- [ ] What is the right cache eviction policy? LRU may not work well with semantic caching because similar queries have overlapping utility.
- [ ] Should we support multi modal ingestion (images, tables) in v1 or defer to v2? Market research says ColPali and page as image approaches are production ready but add significant complexity.

---

## Research References

### Core Math
- Cormack, Clarke, Buettcher (2009): Reciprocal Rank Fusion (RRF)
- Nemhauser, Wolsey, Fisher (1978): Submodular function maximization
- Vovk, Gammerman, Shafer (2005): Conformal Prediction
- Angelopoulos and Bates (2021): A Gentle Introduction to Conformal Prediction

### RAG and Retrieval
- Fang et al. (2024): HippoRAG, knowledge graph augmented retrieval with PPR
- Khattab and Zaharia (2020): ColBERT, late interaction for efficient reranking
- Nogueira and Cho (2019): Cross encoder passage reranking

### Infrastructure
- Subramanya et al. (2019): DiskANN, billion scale approximate nearest neighbors
- Singh et al. (2021): FreshDiskANN, streaming updates to DiskANN

### Market and Competitive
- Vector DB market: $2.5 to $3B in 2026, projected $7 to $15B by 2030 to 2034
- Qdrant Series B: $50M (March 2026), total $87.8M
- Pinecone: ~$750M valuation, ~$26.6M 2025 revenue
- EU AI Act: high risk AI system deadline December 2, 2027

---

## Lessons Learned

- **September 2026**: Python SDK tests complete via `pytest` over `httpx` mock. Node SDK established with `vitest` infrastructure. The core math implementations in `sievrant-math` (RRF, Submodular, Conformal) execute in bare metal microseconds (<10ms for 1000 items), validating AD001 (Rust as primary language).
- **September 2026**: When dealing with WSL to Windows file boundaries in Node.js development, native Node.js binaries from the Windows path will fail when run over UNC paths (`\\wsl.localhost\Ubuntu\...`). Need to ensure Linux `npm` and `node` are used exclusively within the WSL terminal for test execution.

---

## Version History

| Date | Author | Change |
|---|---|---|
| 2026 08 30 | Initial | Project created, all architectural decisions documented |
| 2026 09 09 | Agent | Completed Core SDKs, documented math benchmarks, successfully pushed initial working release to GitHub. |
