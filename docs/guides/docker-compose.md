# Docker Compose Deployment Guide

This guide covers deploying Sievrant using Docker Compose for local development, staging, and production. The Compose file starts three services:

1. **`qdrant`** — Vector database (persistent storage for dense embeddings)
2. **`api`** — Sievrant server (REST on port 8080, gRPC on port 50051)
3. **`otel-collector`** — OpenTelemetry Collector (traces and metrics forwarding)

---

## Prerequisites

- Docker 24+ and Docker Compose v2+ installed
- At least one LLM provider API key (e.g. `OPENAI_API_KEY`) for answer generation
- 4 GB RAM recommended for running all three services

Check your Docker version:

```bash
docker --version          # Must be 24+
docker compose version    # Must be v2+
```

---

## Quick Start (30 seconds)

```bash
# 1. Clone the repository
git clone https://github.com/Nabarup1/sievrant
cd sievrant

# 2. Copy and configure the environment file
cp .env.example .env

# 3. Edit .env and add at least one API key
nano .env   # or: code .env, vim .env

# 4. Start all services
docker compose up -d

# 5. Verify the server is healthy
curl http://localhost:8080/v1/health
```

Expected health response:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 12
}
```

---

## Environment Configuration

The `.env.example` file documents every available variable. At minimum you must set:

```bash
# .env

# LLM provider (at least one required)
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# MISTRAL_API_KEY=...

# Server settings
RUST_LOG=sievrant=info,tower_http=debug
SIEVRANT_QDRANT_URL=http://qdrant:6333

# Conformal prediction defaults (can be overridden per-request)
SIEVRANT_DEFAULT_COVERAGE=0.95
SIEVRANT_DEFAULT_TOKEN_BUDGET=4000
```

### Multi-provider LLM configuration

Sievrant uses `litellm` routing under the hood and respects any provider key you supply:

```bash
OPENAI_API_KEY=sk-...          # GPT-4o, GPT-4o-mini
ANTHROPIC_API_KEY=sk-ant-...   # Claude 3.5 Sonnet, Claude 3 Haiku
MISTRAL_API_KEY=...             # Mistral Large, Mixtral
GROQ_API_KEY=...                # Llama 3 (ultra-low latency)
```

---

## The Docker Compose File

The full `docker-compose.yml` (already in the repo root):

```yaml
services:
  # Qdrant vector database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"    # REST API and dashboard
      - "6334:6334"    # gRPC
    volumes:
      - qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Sievrant API server
  api:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8080:8080"    # REST API
      - "50051:50051"  # gRPC
    env_file:
      - .env
    environment:
      RUST_LOG: "sievrant=info,tower_http=debug"
      SIEVRANT_QDRANT_URL: "http://qdrant:6333"
    depends_on:
      qdrant:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/v1/health"]
      interval: 15s
      timeout: 5s
      retries: 5

  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    volumes:
      - ./otel-collector-config.yml:/etc/otelcol-contrib/config.yaml
    ports:
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
    depends_on:
      - api

volumes:
  qdrant_storage:
```

---

## Common Operations

### View logs

```bash
# All services, follow mode
docker compose logs -f

# Only the API server
docker compose logs -f api

# Only Qdrant
docker compose logs -f qdrant
```

### Stop and start

```bash
# Stop (keeps volumes intact)
docker compose down

# Stop and delete all data (WARNING: deletes your vector data)
docker compose down --volumes

# Restart only the API server after a config change
docker compose restart api
```

### Rebuild after source changes

```bash
# Rebuild the API image (uses cargo-chef caching, only recompiles changed crates)
docker compose build api

# Rebuild and restart in one step
docker compose up -d --build api
```

### Check service health

```bash
docker compose ps
```

Expected output when all services are healthy:

```
NAME                    STATUS              PORTS
sievrant-api-1          running (healthy)   0.0.0.0:8080->8080/tcp
sievrant-qdrant-1       running (healthy)   0.0.0.0:6333->6333/tcp
sievrant-otel-1         running             0.0.0.0:4317->4317/tcp
```

---

## Testing the Full Stack

Once running, verify the full ingestion and query cycle:

### Ingest a document

```bash
curl -X POST http://localhost:8080/v1/ingest \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "corpus": "test",
    "documents": [
      {
        "id": "doc-001",
        "content": "The lease commences January 1, 2025 for a term of 60 months.",
        "metadata": {"filename": "lease.pdf"}
      }
    ]
  }'
```

### Run a query

```bash
curl -X POST http://localhost:8080/v1/query \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When does the lease commence?",
    "corpus": "test",
    "coverage": 0.95
  }'
```

### Use the Python SDK against the Docker stack

```python
import asyncio
from sievrant import SievrantClient

async def main():
    async with SievrantClient(
        base_url="http://localhost:8080",
        api_key="sk-your-key",
    ) as client:
        health = await client.health()
        print(f"Server: {health.status} v{health.version}")

        await client.ingest(
            corpus="test",
            documents=[{"id": "d1", "content": "The penalty for early termination is 3 months rent."}],
        )

        result = await client.query(corpus="test", query="What is the early termination penalty?")
        print(f"Decision: {result.decision}")
        print(f"Answer:   {result.answer}")

asyncio.run(main())
```

---

## Observability Stack

### Qdrant Dashboard

Navigate to `http://localhost:6333/dashboard` in your browser to browse collections, inspect vectors, and monitor storage usage.

### OpenTelemetry

The `otel-collector` service receives traces and metrics from the Sievrant API server via OTLP gRPC on port 4317. By default it outputs to stdout (visible in `docker compose logs otel-collector`).

To forward to a real backend, edit `otel-collector-config.yml`:

```yaml
# otel-collector-config.yml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: "0.0.0.0:4317"
      http:
        endpoint: "0.0.0.0:4318"

exporters:
  # Send to Jaeger
  jaeger:
    endpoint: "jaeger:14250"
    tls:
      insecure: true

  # Or send to Grafana Cloud
  otlphttp:
    endpoint: "https://otlp-gateway-prod-eu-west-0.grafana.net/otlp"
    headers:
      authorization: "Basic <your-base64-token>"

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [jaeger]
    metrics:
      receivers: [otlp]
      exporters: [otlphttp]
```

### Adding Prometheus + Grafana (optional)

Extend the Compose file:

```yaml
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

---

## Production Deployment Considerations

### TLS / HTTPS

For production, put Sievrant behind a reverse proxy (Nginx, Caddy, or Traefik) that handles TLS termination:

```yaml
  # Caddy example (handles TLS automatically via Let's Encrypt)
  caddy:
    image: caddy:2
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
```

`Caddyfile`:

```
api.yourcompany.com {
    reverse_proxy api:8080
}
```

### API Key Management

The default `main.rs` boots with a hardcoded `sk-test` key for development. For production:

1. Set `SIEVRANT_API_KEYS` as a comma-separated list of valid keys, or
2. Connect to a PostgreSQL auth backend (Phase 5 SaaS infrastructure)

Please review our [Security Policy](../../SECURITY.md) for detailed guidance on handling secrets.
In `docker-compose`, prefer using Docker secrets or a `.env` file instead of hardcoding keys in `docker-compose.yml`.

### Resource Limits

Add resource limits to prevent memory pressure:

```yaml
  api:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 2G
        reservations:
          memory: 512M
```

### Data Persistence

The `qdrant_storage` named volume persists your vector data across restarts. For production, back this up regularly:

```bash
# Backup Qdrant data
docker run --rm \
  -v sievrant_qdrant_storage:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/qdrant-$(date +%Y%m%d).tar.gz /data
```

---

## Troubleshooting

### Server fails to start with "connection refused" to Qdrant

Qdrant takes 5-10 seconds to initialise. The Compose `depends_on` healthcheck should handle this, but if you hit it manually:

```bash
docker compose up qdrant -d
# Wait for healthy:
docker compose ps qdrant
# Then start the API:
docker compose up api -d
```

### Port 8080 already in use

```bash
# Find what is using port 8080
lsof -i :8080

# Or change the host port in docker-compose.yml:
ports:
  - "8081:8080"   # Use 8081 on the host instead
```

### AuthError: API key is invalid

The default key for development is `sk-test`. Use:

```bash
curl -H "Authorization: Bearer sk-test" http://localhost:8080/v1/health
```

For production, set your keys via the environment variables described above.

### Cargo build fails inside Docker

The Dockerfile uses `cargo-chef` for layer caching. If you encounter a stale cache:

```bash
docker compose build --no-cache api
```

---

## Next Steps

- [Rust SDK Guide](rust-sdk.md) — Use Sievrant from Rust directly
- [Python SDK Guide](python-sdk.md) — Full async Python client reference
- [REST API reference](../api-reference/rest-api.md) — Full endpoint documentation
- [Architecture deep dive](../concepts/architecture.md) — How the pipeline works
