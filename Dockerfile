# Multi-stage build using cargo-chef for dependency layer caching.
# Only the server binary and its dependencies are compiled: the Python/Node
# binding crates and benchmarks are skipped.

FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json -p sievrant-server
COPY . .
RUN cargo build --release -p sievrant-server --bin sievrant-server

FROM ubuntu:24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /app sievrant \
    && mkdir -p /app/data && chown -R sievrant /app
WORKDIR /app
COPY --from=builder /app/target/release/sievrant-server /usr/local/bin/sievrant-server

USER sievrant
ENV SIEVRANT_DATA_DIR=/app/data \
    SIEVRANT_HOST=0.0.0.0 \
    SIEVRANT_PORT=8080 \
    SIEVRANT_LOG_FORMAT=json \
    RUST_LOG=info,tower_http=warn
VOLUME ["/app/data"]
EXPOSE 8080 50051
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/v1/health || exit 1

CMD ["sievrant-server"]
