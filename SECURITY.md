# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

To report a security issue, email **security@sievrant.dev** with the subject
line `[SECURITY] <brief description>`.

Include as much detail as possible:

- A description of the vulnerability and its potential impact.
- Steps to reproduce or a proof-of-concept.
- Affected versions and components (crate name, SDK, etc.).

You will receive a response within **48 hours** acknowledging receipt.
We aim to provide a fix or mitigation within **14 days** for critical issues
and **30 days** for moderate ones.

## Disclosure Policy

We follow responsible disclosure. Once a fix is released, we will:

1. Publish a GitHub Security Advisory.
2. Credit the reporter in the advisory (unless anonymity is requested).
3. Note the fix in `CHANGELOG.md`.

## Threat Model Summary

Sievrant runs as a server-side retrieval engine. The primary threat surface is:

- **Authentication**: API keys are SHA-256 hashed before comparison inside
  `rest.rs`; raw keys are never stored.
- **Authorization**: Per-request ACL groups are resolved by `AclResolver` and
  enforced at every index query. Users outside a document's ACL groups cannot
  retrieve that document.
- **Input validation**: Query strings and ingest payloads are deserialized
  through typed Rust structs backed by `serde`. Unexpected fields are rejected.
- **Network**: The server binds on a configurable host (default `0.0.0.0:8080`).
  In production deployments, place the server behind a TLS-terminating reverse
  proxy (nginx, Caddy, etc.) and restrict inbound access with firewall rules.
- **Secrets**: All secrets (LLM API keys, Qdrant credentials) are read from
  environment variables. No secret should ever be hard-coded in source.
- **Supply chain**: Dependency versions are pinned via `Cargo.lock`. Dependabot
  is configured to open weekly PRs for updates across Cargo, pip, and npm.

## Secure Configuration Checklist

- [ ] Copy `.env.example` to `.env` and fill in real values.
- [ ] Set `SIEVRANT_API_KEY` to a strong random secret (at least 32 chars).
- [ ] Do not expose port `8080` directly to the internet without a TLS proxy.
- [ ] Rotate `SIEVRANT_API_KEY` periodically and after any suspected leak.
- [ ] Enable `SIEVRANT_AUDIT_ENABLED=true` to maintain a tamper-evident log.
