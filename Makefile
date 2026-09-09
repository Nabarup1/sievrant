# Sievrant Makefile
# Common development commands

.PHONY: build test bench lint format clean run-server run-mcp run-cli setup check

# Build all crates
build:
	cargo build --workspace

# Build in release mode
release:
	cargo build --workspace --release

# Run all tests
test:
	cargo test --workspace

# Run tests with output
test-verbose:
	cargo test --workspace -- --nocapture

# Run property based tests (slower, more thorough)
test-prop:
	cargo test --workspace -- --include-ignored

# Run benchmarks
bench:
	cargo bench --workspace

# Lint with clippy
lint:
	cargo clippy --workspace --all-targets -- -D warnings

# Format code
format:
	cargo fmt --all

# Check formatting without modifying
format-check:
	cargo fmt --all -- --check

# Full check: lint plus format check plus test
check: lint format-check test

# Clean build artifacts
clean:
	cargo clean

# Run the HTTP/MCP server
run-server:
	cargo run -p sievrant-server

# Run the CLI tool
run-cli:
	cargo run -p sievrant-cli -- $(ARGS)

# Watch for changes and rebuild
watch:
	cargo watch -x check -x test

# Watch with clippy
watch-lint:
	cargo watch -x clippy

# Setup: create data directories and check dependencies
setup:
	@echo "Checking Rust toolchain..."
	@rustc --version
	@cargo --version
	@echo ""
	@echo "Creating data directories..."
	@mkdir -p data/wal data/audit data/calibration models
	@echo ""
	@echo "Checking Qdrant..."
	@curl -s http://localhost:6333/healthz > /dev/null 2>&1 && echo "Qdrant is running" || echo "WARNING: Qdrant is not running on localhost:6333"
	@echo ""
	@echo "Checking Python venv..."
	@test -d .venv && echo "Python venv exists" || echo "WARNING: Python venv not found. Run: python3 -m venv .venv"
	@echo ""
	@echo "Setup check complete."

# Generate documentation
doc:
	cargo doc --workspace --no-deps --open

# Run a specific crate's tests
test-crate:
	cargo test -p $(CRATE)

# Run a specific crate's benchmarks
bench-crate:
	cargo bench -p $(CRATE)
