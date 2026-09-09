import { http, HttpResponse } from 'msw';

export const handlers = [
  http.get('http://localhost:8080/v1/health', () => {
    return HttpResponse.json({ status: 'ok', version: '0.1.0' });
  }),

  http.post('http://localhost:8080/v1/query', async ({ request }) => {
    const authHeader = request.headers.get('Authorization');
    if (!authHeader || authHeader !== 'Bearer test-key') {
      return new HttpResponse(null, { status: 401, statusText: 'Unauthorized' });
    }

    const body = (await request.json()) as any;
    if (body.corpus === 'not-found') {
      return new HttpResponse(null, { status: 404, statusText: 'Not Found' });
    }
    if (body.corpus === 'server-error') {
      return new HttpResponse(null, { status: 500, statusText: 'Internal Server Error' });
    }
    if (body.corpus === 'timeout') {
      await new Promise((resolve) => setTimeout(resolve, 500));
      return HttpResponse.json({});
    }

    if (body.query === 'abstain please') {
      return HttpResponse.json({
        query_id: 'test-id',
        decision: 'ABSTAINED',
        answer: null,
        evidence: [],
        coverage_guarantee: 0.0,
        latency_ms: 10,
        route: 'STANDARD',
        cache_hit: false,
        audit: { session_hash: 'abc123', corpus_hash: 'def456', model_used: null, stages_ms: {} },
      });
    }

    return HttpResponse.json({
      query_id: 'test-id',
      decision: 'ANSWERED',
      answer: 'The lease commences January 1, 2025.',
      evidence: [],
      coverage_guarantee: 0.95,
      latency_ms: 42,
      route: 'FAST',
      cache_hit: false,
      audit: {
        session_hash: 'abc123',
        corpus_hash: 'def456',
        model_used: 'gpt-4o-mini',
        stages_ms: {},
      },
    });
  }),

  http.post('http://localhost:8080/v1/ingest', async ({ request }) => {
    const authHeader = request.headers.get('Authorization');
    if (!authHeader || authHeader !== 'Bearer test-key') {
      return new HttpResponse(null, { status: 401, statusText: 'Unauthorized' });
    }
    return HttpResponse.json({ ingested: 1, failed: 0 });
  }),
  
  http.get('http://localhost:8080/v1/corpora', async ({ request }) => {
    const authHeader = request.headers.get('Authorization');
    if (!authHeader || authHeader !== 'Bearer test-key') {
      return new HttpResponse(null, { status: 401, statusText: 'Unauthorized' });
    }
    return HttpResponse.json({ corpora: [{ name: 'test-corpus', document_count: 10, byte_size: 1024, created_at: '2025-01-01T00:00:00Z' }] });
  }),

  http.delete('http://localhost:8080/v1/corpora/:corpus/documents/:id', async ({ request, params }) => {
    const authHeader = request.headers.get('Authorization');
    if (!authHeader || authHeader !== 'Bearer test-key') {
      return new HttpResponse(null, { status: 401, statusText: 'Unauthorized' });
    }
    const id = params.id;
    if (id === 'non-existent') {
      return HttpResponse.json({ deleted: false });
    }
    return HttpResponse.json({ deleted: true });
  })
];
