import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { SievrantClient } from '../client';
import { server } from './mocks/server';
import { AuthError, CorpusNotFoundError, SievrantServerError, SievrantTimeoutError } from '../errors';

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
afterEach(() => {
  server.resetHandlers();
  delete process.env.SIEVRANT_BASE_URL;
  delete process.env.SIEVRANT_API_KEY;
});

describe('SievrantClient', () => {
  describe('Constructor validation', () => {
    it('should throw when base_url is missing', () => {
      expect(() => new SievrantClient({ apiKey: 'test-key' })).toThrow(/base_url/i);
    });

    it('should throw when api_key is missing', () => {
      expect(() => new SievrantClient({ baseUrl: 'http://localhost:8080' })).toThrow(/api_key/i);
    });

    it('should read from environment variables', () => {
      process.env.SIEVRANT_BASE_URL = 'http://env-url:8080';
      process.env.SIEVRANT_API_KEY = 'env-key';
      const client = new SievrantClient();
      // Using an internal property check or just seeing it doesn't throw
      expect(client).toBeDefined();
    });
  });

  describe('health() method', () => {
    it('should return a HealthResponse on 200', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'test-key' });
      const response = await client.health();
      expect(response.status).toBe('ok');
      expect(response.version).toBe('0.1.0');
    });
  });

  describe('query() method', () => {
    it('should return an EvidencePacket on 200 with decision ANSWERED', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'test-key' });
      const response = await client.query({ corpus: 'test-corpus', query: 'What is the lease date?' });
      expect(response.decision).toBe('ANSWERED');
      expect(response.answer).toBe('The lease commences January 1, 2025.');
      expect(response.coverage_guarantee).toBe(0.95);
    });

    it('should return an EvidencePacket on 200 with decision ABSTAINED', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'test-key' });
      const response = await client.query({ corpus: 'test-corpus', query: 'abstain please' });
      expect(response.decision).toBe('ABSTAINED');
      expect(response.answer).toBeNull();
    });

    it('should throw AuthError on 401', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'wrong-key' });
      await expect(client.query({ corpus: 'test-corpus', query: 'query' })).rejects.toThrow(AuthError);
    });

    it('should throw CorpusNotFoundError on 404', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'test-key' });
      await expect(client.query({ corpus: 'not-found', query: 'query' })).rejects.toThrow(CorpusNotFoundError);
    });

    it('should throw SievrantServerError on 500', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'test-key', maxRetries: 0 });
      await expect(client.query({ corpus: 'server-error', query: 'query' })).rejects.toThrow(SievrantServerError);
    });

    it('should throw SievrantTimeoutError when the request exceeds timeout', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'test-key', timeoutMs: 100, maxRetries: 0 });
      await expect(client.query({ corpus: 'timeout', query: 'query' })).rejects.toThrow(SievrantTimeoutError);
    });
  });

  describe('ingest() method', () => {
    it('should return IngestResponse on 200', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'test-key' });
      const response = await client.ingest({ corpus: 'test-corpus', documents: [{ id: '1', content: 'doc1' }] });
      expect(response.ingested).toBe(1);
      expect(response.failed).toBe(0);
    });
  });

  describe('listCorpora() method', () => {
    it('should return an array of CorpusMetadata', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'test-key' });
      const corpora = await client.listCorpora();
      expect(corpora).toHaveLength(1);
      expect(corpora[0].name).toBe('test-corpus');
      expect(corpora[0].document_count).toBe(10);
    });
  });

  describe('deleteDocument() method', () => {
    it('should return true when server returns deleted: true', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'test-key' });
      const result = await client.deleteDocument('test-corpus', 'doc-123');
      expect(result).toBe(true);
    });

    it('should return false when server returns deleted: false', async () => {
      const client = new SievrantClient({ baseUrl: 'http://localhost:8080', apiKey: 'test-key' });
      const result = await client.deleteDocument('test-corpus', 'non-existent');
      expect(result).toBe(false);
    });
  });
});
