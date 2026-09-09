/**
 * SievrantClient — async TypeScript client for the Sievrant server.
 *
 * Uses native fetch (Node 18+) with automatic retries, timeout handling,
 * and clean error mapping to typed SDK exceptions.
 */

import type {
  EvidencePacket,
  HealthResponse,
  IngestResponse,
  IngestOptions,
  QueryOptions,
  SievrantClientOptions,
  CorpusMetadata,
} from './types.js';

import {
  AuthError,
  CorpusNotFoundError,
  SievrantServerError,
  SievrantTimeoutError,
  RateLimitError,
  SievrantError,
} from './errors.js';

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_RETRIES = 3;
const SDK_VERSION = '0.1.0';

export class SievrantClient {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly extraHeaders: Record<string, string>;

  constructor(options: SievrantClientOptions = {}) {
    const baseUrl = options.baseUrl ?? process.env['SIEVRANT_BASE_URL'];
    const apiKey = options.apiKey ?? process.env['SIEVRANT_API_KEY'];

    if (!baseUrl) {
      throw new Error(
        'baseUrl is required. Pass it in options or set SIEVRANT_BASE_URL env var.'
      );
    }
    if (!apiKey) {
      throw new Error(
        'apiKey is required. Pass it in options or set SIEVRANT_API_KEY env var.'
      );
    }

    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
    this.extraHeaders = options.headers ?? {};
  }

  /** Check server health. No authentication required. */
  async health(): Promise<HealthResponse> {
    return this.get<HealthResponse>('/v1/health', { auth: false });
  }

  /**
   * Execute a retrieval query against a corpus.
   *
   * @param options - Query options including corpus, query text, and configuration.
   * @returns EvidencePacket with decision, answer, evidence, and audit block.
   */
  async query(options: QueryOptions): Promise<EvidencePacket> {
    const endpoint = options.full ? '/v1/query/full' : '/v1/query';
    const body = {
      query: options.query,
      corpus: options.corpus,
      coverage: options.coverage ?? 0.95,
      token_budget: options.tokenBudget ?? 4000,
      acl_groups: options.aclGroups ?? [],
      skip_cache: options.skipCache ?? false,
      route: options.route ?? 'AUTO',
    };
    return this.post<EvidencePacket>(endpoint, body);
  }

  /**
   * Ingest documents into a corpus.
   * The corpus is created automatically if it does not exist.
   */
  async ingest(options: IngestOptions): Promise<IngestResponse> {
    const body = {
      corpus: options.corpus,
      documents: options.documents,
    };
    return this.post<IngestResponse>('/v1/ingest', body);
  }

  /** List all corpora accessible to this API key. */
  async listCorpora(): Promise<CorpusMetadata[]> {
    const data = await this.get<{ corpora: CorpusMetadata[] }>('/v1/corpora');
    return data.corpora;
  }

  /** Get metadata for a specific corpus. */
  async getCorpus(corpus: string): Promise<CorpusMetadata> {
    return this.get<CorpusMetadata>(`/v1/corpora/${encodeURIComponent(corpus)}`);
  }

  /** Delete a document from a corpus. */
  async deleteDocument(corpus: string, docId: string): Promise<boolean> {
    const data = await this.delete_<{ deleted: boolean }>(
      `/v1/corpora/${encodeURIComponent(corpus)}/documents/${encodeURIComponent(docId)}`
    );
    return data.deleted;
  }

  private buildHeaders(auth = true): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'User-Agent': `sievrant-node/${SDK_VERSION}`,
      ...this.extraHeaders,
    };
    if (auth) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  private async get<T>(path: string, opts: { auth?: boolean } = {}): Promise<T> {
    return this.request<T>('GET', path, undefined, opts.auth ?? true);
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>('POST', path, body, true);
  }

  private async delete_<T>(path: string): Promise<T> {
    return this.request<T>('DELETE', path, undefined, true);
  }

  private async request<T>(
    method: string,
    path: string,
    body: unknown,
    auth: boolean,
    attempt = 0
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    let response: Response;
    try {
      response = await fetch(url, {
        method,
        headers: this.buildHeaders(auth),
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } catch (err: unknown) {
      clearTimeout(timer);
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw new SievrantTimeoutError(this.timeoutMs);
      }
      throw new SievrantError(`Network error: ${String(err)}`);
    } finally {
      clearTimeout(timer);
    }

    if (response.ok) {
      return response.json() as Promise<T>;
    }

    // Retry on 5xx errors
    const shouldRetry = response.status >= 500 && attempt < this.maxRetries;
    if (shouldRetry) {
      const delay = 1000 * Math.pow(1.5, attempt);
      await new Promise((r) => setTimeout(r, delay));
      return this.request<T>(method, path, body, auth, attempt + 1);
    }

    let errorBody: Record<string, unknown> = {};
    try {
      errorBody = (await response.json()) as Record<string, unknown>;
    } catch {
      // ignore JSON parse failure on error responses
    }

    const message = (errorBody['message'] as string) ?? response.statusText;
    const errorCode = errorBody['error'] as string | undefined;

    switch (response.status) {
      case 401:
        throw new AuthError(message);
      case 404:
        throw new CorpusNotFoundError((errorBody['corpus'] as string) ?? 'unknown');
      case 429: {
        const retryAfter = response.headers.get('Retry-After');
        throw new RateLimitError(retryAfter ? parseInt(retryAfter, 10) : undefined);
      }
      default:
        if (response.status >= 500) {
          throw new SievrantServerError(message, response.status, errorCode);
        }
        throw new SievrantError(message, response.status);
    }
  }
}
