# Sievrant: Setup Guide

Step by step instructions for setting up the development environment from scratch.

---

## Prerequisites

### Operating System
- Linux (Ubuntu 22.04+ recommended) or macOS
- Windows via WSL2 (tested on Ubuntu under WSL)

### Required Tools
- Git
- curl or wget
- A C/C++ compiler (gcc, clang) for native dependencies
- pkg-config and libssl-dev (for reqwest TLS)
- protobuf-compiler (for gRPC, if using the server)

---

## 1. Install Rust Toolchain

```bash
# Install rustup (the Rust toolchain manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the prompts, choose default installation
# Then reload your shell
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version

# Install additional components
rustup component add clippy rustfmt

# Optional: install cargo-watch for auto rebuild during development
cargo install cargo-watch
```

---

## 2. Install System Dependencies

### Ubuntu / Debian / WSL
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    cmake \
    git \
    curl
```

### macOS
```bash
brew install openssl protobuf cmake pkg-config
```

---

## 3. Install and Run Qdrant (Vector Database)

Sievrant uses Qdrant for the vector index layer. You need a running Qdrant instance for development.

### Option A: Docker (recommended)
```bash
docker pull qdrant/qdrant:latest
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### Option B: Binary
```bash
# Download the latest binary from https://github.com/qdrant/qdrant/releases
# Extract and run
./qdrant
```

### Verify Qdrant is running
```bash
curl http://localhost:6333/healthz
# Should return: {"title":"qdrant ...","version":"..."}
```

---

## 4. Download ONNX Reranker Model (Optional)

If using local cross encoder reranking (default config):

```bash
mkdir -p models

# Download BGE Reranker v2 M3 ONNX model
# Option 1: From Hugging Face (requires huggingface_hub[cli])
pip install -U "huggingface_hub[cli]"
hf download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3

# Option 2: Manual download
# Visit https://huggingface.co/BAAI/bge-reranker-v2-m3
# Download the ONNX model file to models/
```

If you prefer API based reranking (Cohere), set `SIEVRANT_RERANKER_TYPE=cohere` in your `.env` and provide a `COHERE_API_KEY`.

---

## 5. Set Up Python Environment (for tooling)

Python is used only for benchmarking, evaluation, and calibration data preparation. It is not required for running Sievrant itself.

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify
python -c "import ragas; print('ragas version:', ragas.__version__)"
```

---

## 6. Configure Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and fill in your API keys
# At minimum you need one LLM provider key (OPENAI_API_KEY or ANTHROPIC_API_KEY)
nano .env
```

---

## 7. Build the Project

```bash
# Build all crates in the workspace
cargo build

# Build in release mode (for benchmarking)
cargo build --release

# Check that everything compiles without warnings
cargo clippy --workspace --all-targets

# Format check
cargo fmt --all -- --check
```

---

## 8. Run Tests

```bash
# Run all unit tests
cargo test --workspace

# Run tests with output
cargo test --workspace -- --nocapture

# Run a specific crate's tests
cargo test -p sievrant-math

# Run property based tests (may take longer)
cargo test --workspace -- --include-ignored
```

---

## 9. Run Benchmarks

```bash
# Run all benchmarks
cargo bench --workspace

# Run benchmarks for a specific crate
cargo bench -p sievrant-math

# Results will be in target/criterion/
```

---

## 10. Run the Server (when implemented)

```bash
# Run the HTTP server
cargo run -p sievrant-server

# Run with specific log level
RUST_LOG=debug cargo run -p sievrant-server

# Run the CLI tool
cargo run -p sievrant-cli -- --help
```

---

## 11. Development Workflow

```bash
# Watch for changes and rebuild automatically
cargo watch -x check -x test

# Run clippy on save
cargo watch -x clippy

# Format code
cargo fmt --all
```

---

## Directory Structure After Setup

```
sievrant/
  .env                    # your local env config
  .env.example            # reference template
  .gitignore
  Cargo.toml              # workspace manifest
  Makefile
  PRD.md
  TRD.md
  SYSTEM_ARCHITECTURE.md
  SETUP.md                # this file
  memory.md               # living project memory
  requirements.txt        # Python tooling deps
  prompts/                # code generation prompts (30 files)
  crates/                 # Rust source crates (15 crates)
  models/                 # ONNX models (not in git)
  data/                   # runtime data (not in git)
    wal/
    audit/
    calibration/
  .venv/                  # Python venv (not in git)
  qdrant_storage/         # local Qdrant data (not in git)
```
