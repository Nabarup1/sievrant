# Multi-stage build using cargo-chef for aggressive layer caching.
# Stage 1 caches dependency compilation independently from application code.
# This means only re-compiling changed crates rather than the full dependency tree.

FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /app
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

# Stage 2: Generate a dependency recipe (a manifest of all crates and their lock files).
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Stage 3: Compile only the dependencies using the recipe.
# This layer is cached across all builds as long as Cargo.lock and Cargo.toml do not change.
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Now copy the actual source code and build the application binary.
COPY . .
RUN cargo build --release --bin sievrant-server

# Stage 4: Minimal runtime image. Copies only the compiled binary.
# Using ubuntu:24.04 to ensure we have a compatible glibc version (2.39).
FROM ubuntu:24.04 AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/sievrant-server /usr/local/bin/sievrant-server

ENV RUST_LOG=sievrant=info
EXPOSE 8080

CMD ["sievrant-server"]
