/**
 * LangChain.js integration for Sievrant.
 *
 * Provides SievrantRetriever implementing BaseRetriever from @langchain/core.
 * Drop-in replacement for any LangChain.js retriever.
 */

import { SievrantClient } from '../client.js';
import type { EvidenceChunk, QueryOptions } from '../types.js';

let BaseRetriever: typeof import('@langchain/core/retrievers').BaseRetriever;
let Document: typeof import('@langchain/core/documents').Document;

try {
  const retrievers = await import('@langchain/core/retrievers');
  const documents = await import('@langchain/core/documents');
  BaseRetriever = retrievers.BaseRetriever;
  Document = documents.Document;
} catch {
  throw new Error(
    'LangChain integration requires @langchain/core. Install it with: npm install @langchain/core'
  );
}

function chunkToDocument(chunk: EvidenceChunk): InstanceType<typeof Document> {
  return new Document({
    pageContent: chunk.text,
    metadata: {
      chunk_id: chunk.chunk_id,
      source_doc: chunk.source_doc,
      doc_version_hash: chunk.doc_version_hash,
      path: chunk.path.join(' > '),
      page: chunk.page,
      char_span: chunk.char_span,
      retriever: chunk.retriever,
      rrf_score: chunk.rrf_score,
      rerank_score: chunk.rerank_score,
      nonconformity_score: chunk.nonconformity_score,
      submodular_marginal_gain: chunk.submodular_marginal_gain,
      acl_groups: chunk.acl_groups,
      freshness_hours: chunk.freshness_hours,
      token_count: chunk.token_count,
      ingested_at: chunk.ingested_at,
    },
  });
}

export interface SievrantRetrieverOptions {
  baseUrl: string;
  apiKey: string;
  corpus: string;
  coverage?: number;
  tokenBudget?: number;
  k?: number;
  aclGroups?: string[];
  skipCache?: boolean;
  route?: QueryOptions['route'];
  timeoutMs?: number;
}

export class SievrantRetriever extends BaseRetriever {
  lc_namespace = ['sievrant', 'retrievers'];

  private readonly options: SievrantRetrieverOptions;

  constructor(options: SievrantRetrieverOptions) {
    super();
    this.options = options;
  }

  async _getRelevantDocuments(
    query: string
  ): Promise<InstanceType<typeof Document>[]> {
    const client = new SievrantClient({
      baseUrl: this.options.baseUrl,
      apiKey: this.options.apiKey,
      timeoutMs: this.options.timeoutMs,
    });

    const packet = await client.query({
      corpus: this.options.corpus,
      query,
      coverage: this.options.coverage ?? 0.95,
      tokenBudget: this.options.tokenBudget ?? 4000,
      aclGroups: this.options.aclGroups,
      skipCache: this.options.skipCache,
      route: this.options.route,
      full: true,
    });

    let chunks = packet.evidence;
    if (this.options.k !== undefined) {
      chunks = chunks.slice(0, this.options.k);
    }

    const docs = chunks.map(chunkToDocument);

    // Attach top-level packet metadata
    for (const doc of docs) {
      doc.metadata['sievrant_decision'] = packet.decision;
      doc.metadata['sievrant_query_id'] = packet.query_id;
      doc.metadata['sievrant_coverage_guarantee'] = packet.coverage_guarantee;
      doc.metadata['sievrant_route'] = packet.route;
      doc.metadata['sievrant_cache_hit'] = packet.cache_hit;
      doc.metadata['sievrant_session_hash'] = packet.audit.session_hash;
    }

    return docs;
  }
}
