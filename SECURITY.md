# Security Policy

## Supported Versions

Currently, Sievrant is in early development. Only the latest release on the `main` branch is supported for security updates. 

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Threat Model

Sievrant's security architecture is built on the following assumptions:

1.  **API Tokens**: Clients must use a valid Bearer token for all write (ingest) and read (query) operations. Tokens are managed by the server and are hashed using SHA-256 for validation.
2.  **Access Control**: Documents can be ingested with associated ACL groups (e.g., `["engineering", "hr"]`). During a query, if the user token maps to specific groups, only documents intersecting with those groups are returned.
3.  **No Internal Auth**: The server currently does not implement a built-in user management system. It relies on tokens issued externally or configured via environment variables.

### Key Handling

-   **Secret Storage**: Do NOT store bare API keys in source control or plain text configuration files. Use environment variables (e.g., `.env`) or a secrets manager like AWS Secrets Manager, HashiCorp Vault, or Doppler.
-   **Docker Compose**: When using `docker-compose`, inject secrets at runtime or mount them securely as environment variables rather than hardcoding them in the `docker-compose.yml`.

## Reporting a Vulnerability

If you discover a security vulnerability within Sievrant, please do not disclose it publicly. Instead, send an email to the maintainers (or open a confidential GitHub Security Advisory if available on the repository).

We will review your report and respond with an initial assessment within 48 hours. Please provide the following details:
- A description of the vulnerability and its impact.
- Steps to reproduce the issue.
- Potential mitigation or suggestions (optional).
