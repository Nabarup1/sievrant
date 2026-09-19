# Sievrant Node.js SDK

The official TypeScript/Node.js SDK for Sievrant — the statistically audited retrieval decision engine.

Sievrant provides a Conformal Prediction layer over vector databases, graphs, and search engines to give statistical guarantees on retrieval coverage and precision.

## Installation

```bash
npm install sievrant
```

## Usage

### Basic Retrieval

```typescript
import { SievrantClient } from 'sievrant';

const client = new SievrantClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'sk-your-key', // Optional if authentication is disabled
});

async function run() {
  const packet = await client.query({
    corpus: 'my_corpus',
    query: 'What are the benefits of conformal prediction?',
    coverage: 0.95, // 95% statistical guarantee
  });

  console.log(`Decision: ${packet.decision}`);
  if (packet.decision !== 'ABSTAINED') {
    packet.evidence.forEach(chunk => {
      console.log(`Found chunk from ${chunk.source_doc} (Score: ${chunk.rrf_score})`);
      console.log(chunk.text);
    });
  }
}

run();
```

## Integrations

Sievrant natively integrates with LangChain and Vercel AI SDK.

### LangChain

```typescript
import { SievrantRetriever } from 'sievrant/integrations/langchain';

const retriever = new SievrantRetriever({
  baseUrl: 'http://localhost:8080',
  apiKey: 'sk-your-key',
  corpus: 'my_corpus',
  coverage: 0.95,
});

const docs = await retriever.invoke('How does Sievrant work?');
```

## Testing

The test suite uses `vitest` and `msw` to test the SDK completely offline without requiring a live Sievrant server instance.

```bash
npm run test
```
