/**
 * Main entry point for the Sievrant Node.js SDK.
 * Re-exports all public types, errors, and the client.
 */

export { SievrantClient } from './client.js';

export {
  SievrantError,
  AuthError,
  CorpusNotFoundError,
  SievrantServerError,
  SievrantTimeoutError,
  RateLimitError,
} from './errors.js';

export type {
  EvidencePacket,
  EvidenceChunk,
  AuditBlock,
  RetrievalStagesMs,
  HealthResponse,
  IngestResponse,
  IngestOptions,
  IngestDocument,
  QueryOptions,
  SievrantClientOptions,
  CorpusMetadata,
  Decision,
  Route,
  RetrieverType,
} from './types.js';
