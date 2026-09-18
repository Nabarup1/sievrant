# Sievrant Security & Threat Model

This document outlines the security posture, threat model, and handling of secrets within the Sievrant RAG engine.

## 1. Secrets Management

Sievrant relies on sensitive credentials to authenticate with backing stores and LLM providers. **Under no circumstances should any secrets be hardcoded or checked into source control.**

- **API Keys**: Keys such as `SIEVRANT_API_KEY`, `QDRANT_API_KEY`, and `OPENAI_API_KEY` must be injected into the application exclusively via environment variables or a dedicated secret manager (e.g., AWS Secrets Manager, HashiCorp Vault) at runtime.
- **Environment Files**: The provided `.env.example` file is for documentation purposes only. Real `.env` files containing secrets are ignored via `.gitignore` and must never be committed.

## 2. API Authentication & Token Hashing

To protect against timing attacks and database compromise, Sievrant implements a robust API key verification scheme:

- **Verification (`AppState::verify_token`)**: Incoming API requests are authenticated by comparing the provided token against the configured server key.
- **Hashing Strategy**: Instead of performing plain-text comparisons, tokens are cryptographically hashed using **SHA-256** prior to validation. This ensures that even if application memory or logs are dumped, the raw API tokens cannot be easily recovered or reused.

## 3. Tenant Isolation & Access Control (ACL)

Sievrant supports multi-tenant document retrieval, ensuring that users only have access to documents they are authorized to see.

- **ACL Groups**: Every chunk of evidence ingested into Sievrant is tagged with an optional list of `acl_groups` in its `ChunkMetadata`.
- **Query Time Filtering**: During retrieval, queries are accompanied by the user's `acl_groups`. The retrieval engine (e.g., `TreeIndex::beam_search`, Vector Search) applies a hard filter to only consider candidate chunks where at least one ACL group matches. 
- **Isolation Boundary**: This mechanism guarantees a strict isolation boundary between tenants. A malicious user cannot retrieve or reconstruct documents belonging to another tenant's ACL group, as the filter is enforced natively at the index traversal layer before the LLM synthesizes an answer.

## 4. Reporting Vulnerabilities

If you discover a security vulnerability within Sievrant, please do not disclose it publicly. Instead, submit a report to the maintainers directly. We take all security concerns seriously and will work to address them promptly.
