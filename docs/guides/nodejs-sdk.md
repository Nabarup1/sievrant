# Node.js / TypeScript SDK Guide

The Sievrant Node.js SDK is a fully-typed TypeScript client for the Sievrant server. It works in Node.js 18+ and supports integrations with LangChain.js and Vercel AI SDK.

## Installation

```bash
npm install sievrant
# or
pnpm add sievrant
# or
yarn add sievrant
```

TypeScript 5.0+ is supported. The package ships with its own type definitions — no `@types/sievrant` needed.

---

## Basic Usage

```typescript
import { SievrantClient } from 'sievrant';

const client = new SievrantClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'sk-your-key',
});

// Health check
const health = await client.health();
console.log(`Sievrant ${health.version} — ${health.status}`);

// Ingest documents
await client.ingest({
  corpus: 'legal-docs',
  documents: [
    {
      id: 'contract-001',
      content: 'The lease commences January 1, 2025 for 60 months.',
      metadata: { filename: 'lease.pdf', author: 'Legal Dept' },
    },
  ],
});

// Query
const result = await client.query({
  corpus: 'legal-docs',
  query: 'When does the lease commence?',
  coverage: 0.95,
});

console.log(result.decision);     // "ANSWERED" | "ABSTAINED" | "EVIDENCE_ONLY"
console.log(result.answer);       // string | null
console.log(result.latencyMs);    // number
console.log(result.evidence);     // EvidenceChunk[]
```

---

## Client Options

```typescript
const client = new SievrantClient({
  baseUrl: 'http://localhost:8080',   // Sievrant server URL (required)
  apiKey: 'sk-your-key',             // Bearer token (required)
  timeoutMs: 30_000,                  // Request timeout (default 30000ms)
  maxRetries: 3,                      // Retry on 5xx errors (default 3)
  retryDelay: 1000,                   // Base retry delay in ms (default 1000)
  headers: {                          // Additional HTTP headers
    'X-Custom-Header': 'value',
  },
});
```

**Environment variables** (used when options are omitted):

```bash
SIEVRANT_BASE_URL=http://localhost:8080
SIEVRANT_API_KEY=sk-your-key
```

---

## `query()` Reference

```typescript
import type { QueryOptions, EvidencePacket } from 'sievrant';

const result: EvidencePacket = await client.query({
  corpus: 'legal-docs',           // Required
  query: 'What is the rent?',     // Required
  coverage: 0.95,                 // Coverage target (default 0.95)
  tokenBudget: 4000,              // Max tokens across evidence (default 4000)
  aclGroups: ['legal', 'hr'],     // ACL groups for this user
  skipCache: false,               // Skip semantic cache
  route: 'AUTO',                  // 'AUTO' | 'FAST' | 'STANDARD' | 'SLOW'
});
```

---

## `ingest()` Reference

```typescript
await client.ingest({
  corpus: 'legal-docs',
  documents: [
    {
      id: 'doc-001',              // Required: unique ID
      content: 'Full text...',   // Required: document text
      metadata: {                 // Optional
        filename: 'lease.pdf',
        author: 'Legal Team',
      },
      aclGroups: ['legal'],       // Optional: access control
    },
  ],
});
```

---

## TypeScript Types

```typescript
import type {
  EvidencePacket,
  EvidenceChunk,
  AuditBlock,
  RetrievalStagesMs,
  Decision,
  Route,
  RetrieverType,
} from 'sievrant';

// Decision enum
type Decision = 'ANSWERED' | 'ABSTAINED' | 'EVIDENCE_ONLY';

// Evidence chunk
interface EvidenceChunk {
  chunkId: string;
  sourceDoc: string;
  docVersionHash: string;
  path: string[];
  page?: number;
  charSpan?: [number, number];
  text: string;
  retriever: RetrieverType;
  rrfScore: number;
  rerankScore?: number;
  nonconformityScore: number;
  submodularMarginalGain: number;
  ingestedAt: string;   // ISO 8601
  indexGeneration: number;
  aclGroups: string[];
  freshnessHours: number;
  tokenCount: number;
}

// Full evidence packet
interface EvidencePacket {
  queryId: string;
  traceId?: string;
  decision: Decision;
  answer?: string;
  coverageGuarantee: number;
  empiricalCoverageLast30d?: number;
  route: Route;
  cacheHit: boolean;
  latencyMs: number;
  costUsd?: number;
  evidence: EvidenceChunk[];
  audit: AuditBlock;
}
```

---

## LangChain.js Integration

```typescript
import { SievrantRetriever } from 'sievrant/integrations/langchain';
import { ChatOpenAI } from '@langchain/openai';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const retriever = new SievrantRetriever({
  baseUrl: 'http://localhost:8080',
  apiKey: 'sk-your-key',
  corpus: 'legal-docs',
  coverage: 0.95,
  k: 5,
});

const llm = new ChatOpenAI({ model: 'gpt-4o' });

const prompt = ChatPromptTemplate.fromTemplate(`
Answer based only on this context:
{context}

Question: {input}
`);

const documentChain = await createStuffDocumentsChain({ llm, prompt });
const chain = await createRetrievalChain({
  retriever,
  combineDocsChain: documentChain,
});

const result = await chain.invoke({ input: 'What is the monthly rent?' });
console.log(result.answer);
```

---

## Vercel AI SDK Integration

```typescript
import { SievrantRetriever } from 'sievrant/integrations/vercel-ai';
import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

const retriever = new SievrantRetriever({
  baseUrl: 'http://localhost:8080',
  apiKey: 'sk-your-key',
  corpus: 'legal-docs',
});

// In a Next.js route handler or API
export async function POST(req: Request) {
  const { messages } = await req.json();
  const query = messages[messages.length - 1].content;

  const evidence = await retriever.retrieve(query);

  const result = await streamText({
    model: openai('gpt-4o'),
    system: `Answer using this context:\n${evidence.map(e => e.pageContent).join('\n')}`,
    messages,
  });

  return result.toDataStreamResponse();
}
```

---

## Error Handling

```typescript
import {
  SievrantError,
  AuthError,
  CorpusNotFoundError,
  SievrantServerError,
  SievrantTimeoutError,
} from 'sievrant';

try {
  const result = await client.query({ corpus: 'missing', query: 'test' });
} catch (err) {
  if (err instanceof CorpusNotFoundError) {
    console.error(`Corpus not found: ${err.corpus}`);
  } else if (err instanceof AuthError) {
    console.error('Invalid API key');
  } else if (err instanceof SievrantTimeoutError) {
    console.error('Request timed out');
  } else if (err instanceof SievrantError) {
    console.error(`Sievrant error: ${err.message}`);
  }
}
```

---

## Streaming (gRPC / SSE)

For streaming retrieval (progressive evidence as chunks are ranked), use the stream method:

```typescript
const stream = client.queryStream({
  corpus: 'legal-docs',
  query: 'What are all the payment terms?',
});

for await (const event of stream) {
  if (event.type === 'evidence_chunk') {
    console.log(`Chunk: ${event.chunk.text}`);
  } else if (event.type === 'answer_token') {
    process.stdout.write(event.token);
  } else if (event.type === 'done') {
    console.log(`\nDecision: ${event.packet.decision}`);
  }
}
```

---

## CommonJS Support

The package ships both ESM and CJS builds:

```javascript
// CommonJS
const { SievrantClient } = require('sievrant');

// ESM
import { SievrantClient } from 'sievrant';
```
