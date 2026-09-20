# Security Policy

## Supported Versions

Currently, only the latest `main` branch and the latest stable release are supported with security updates. 

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Instead, please send an email to `security@sievrant.com` with a detailed description of the vulnerability, steps to reproduce, and any potential impact. We will respond within 48 hours to acknowledge the report and provide an estimated timeline for a fix. 

## Threat Model

Sievrant is a retrieval-augmented generation (RAG) engine. As such, it operates on user-provided data and queries, and communicates with external LLM providers. 

### Trust Boundaries

1. **Client to Server**: The boundary between the API client (user) and the Sievrant server. Authenticated via API keys or OAuth (configurable).
2. **Server to Storage**: The boundary between the Sievrant server and the persistent storage (vector indexes, disk). We assume the storage layer is trusted and protected by OS-level access controls.
3. **Server to LLM**: The boundary between the Sievrant server and external LLM APIs (e.g., OpenAI, Anthropic). We assume the LLM provider is a trusted third-party, but we do not blindly trust the generated output to be safe for execution (e.g. XSS if rendered).

### Key Threats Mitigated

* **Prompt Injection**: LLMs are susceptible to prompt injection. Sievrant mitigates this by strictly separating the retrieved evidence from the system prompt, and escaping user queries. However, users should remain vigilant about downstream usage of generated answers.
* **Data Exfiltration**: The conformal gate ensures that only relevant, high-quality chunks are passed to the LLM. The ACL system ensures that users can only retrieve chunks they are authorized to access, preventing cross-tenant data leakage.
* **Denial of Service**: Expensive vector searches are bounded by strict `search_limit` defaults and token budgets to prevent resource exhaustion attacks.

### Remaining Risks

* **LLM Hallucinations**: While reciprocal rank fusion and conformal prediction reduce hallucinations by ensuring high-quality evidence, no RAG system can completely eliminate the risk of an LLM generating false information.
* **Malicious Ingestion**: If an attacker can ingest maliciously crafted documents, they could attempt to influence the retrieval results (e.g., data poisoning). Input validation is performed during ingestion, but administrators should restrict ingestion access to trusted parties.
