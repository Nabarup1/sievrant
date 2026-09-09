# Contributing to Sievrant

Thank you for your interest in contributing to Sievrant. This document covers the contribution process, development setup, and code standards.

---

## Getting Started

### Prerequisites

- Rust 1.78+ (`rustup update stable`)
- Docker and Docker Compose v2
- Python 3.10+ (for SDK development)
- Node.js 18+ (for SDK development)

### Repository Setup

```bash
git clone https://github.com/Nabarup1/sievrant
cd sievrant

# Start supporting infrastructure (Qdrant, Jaeger)
docker-compose up -d qdrant jaeger

# Copy environment template
cp .env.example .env
# Edit .env: add your OPENAI_API_KEY (or ANTHROPIC_API_KEY or GEMINI_API_KEY)
```

### Build

```bash
# Build all crates
cargo build

# Run all Rust tests
cargo test

# Run specific crate tests
cargo test -p sievrant-math
cargo test -p sievrant-server

# Run the server locally
cargo run -p sievrant-server
```

---

## Project Structure

```
sievrant/
  crates/
    sievrant-core/          # Shared types (EvidencePacket, etc.)
    sievrant-math/          # RRF, Submodular, Conformal
    sievrant-index/         # Vector/BM25/Tree/Graph indexes
    sievrant-rerank/        # Cross-encoder rerankers
    sievrant-llm/           # LLM provider clients
    sievrant-query/         # Query processing, agentic loop
    sievrant-orchestrator/  # 10-stage pipeline
    sievrant-storage/       # WAL, LSM, disk storage
    sievrant-acl/           # Access control
    sievrant-calibration/   # Conformal calibration
    sievrant-audit/         # Audit block generation
    sievrant-observability/ # OpenTelemetry, metering
    sievrant-server/        # REST + gRPC server
    sievrant-cli/           # CLI tool
    sievrant/               # Public facade
  sdks/
    python/                 # Python SDK
    nodejs/                 # Node.js/TypeScript SDK
  docs/                     # Documentation
  prompts/                  # Implementation prompts archive
```

---

## Development Workflow

### Running the test suite

```bash
# All Rust tests
cargo test

# Python SDK tests
cd sdks/python
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v

# Node.js SDK tests
cd sdks/nodejs
npm install
npm test
```

### Running benchmarks

```bash
cargo bench -p sievrant-math
# Results appear in target/criterion/
```

---

## Code Standards

### Rust

- All public items must have `///` doc comments with at minimum one sentence describing what the item is.
- Use `tracing::{info, debug, warn, error}` for all logging. Never use `println!` or `eprintln!` in library code.
- Do not use `.unwrap()` in library code. Use `?` propagation or `.expect("descriptive reason")` only where a panic is genuinely correct behavior.
- Run `cargo clippy -- -D warnings` before submitting. All warnings must be resolved.
- Run `cargo fmt` to format code.

### Comments

- Write comments as plain English sentences.
- Comments explain *why*, not *what*. The code says what. The comment says why.

### Python SDK

- All public functions and classes must have Google-style docstrings.
- Type annotations on all function signatures.
- Run `ruff check` and `mypy sievrant/` before submitting.

### Node.js SDK

- All public exports must have JSDoc comments.
- No `any` types unless unavoidable, with a comment explaining why.
- Run `tsc --noEmit` before submitting.

---

## Submitting a Pull Request

1. **Open an issue first** for non-trivial changes (new features, breaking changes). Discuss the approach before implementing.
2. **Create a branch** from `main`: `git checkout -b feat/my-feature`
3. **Write tests** for your changes. PRs without tests for new behavior will not be merged.
4. **Update documentation** in `docs/` if you add or change public APIs.
5. **Update CHANGELOG.md** under `[Unreleased]`.
6. **Open a PR** with a clear description of what changed and why.

---

## Architecture Decisions

Before proposing major changes to the core pipeline, read the academic references in the concept docs:

- `docs/concepts/conformal-prediction.md` — before changing the conformal gate
- `docs/concepts/submodular-selection.md` — before changing evidence selection
- `docs/concepts/architecture.md` — before changing the pipeline stages

These are not arbitrary design choices. They are grounded in peer-reviewed research with provable guarantees.

---

## License

By contributing to Sievrant, you agree that your contributions will be licensed under the Apache 2.0 License.
