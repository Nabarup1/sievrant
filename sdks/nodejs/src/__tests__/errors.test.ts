import { describe, it, expect } from 'vitest';
import { SievrantError, AuthError, CorpusNotFoundError, SievrantServerError, SievrantTimeoutError } from '../errors';

describe('Error classes', () => {
  it('AuthError has status 401 and extends SievrantError', () => {
    const error = new AuthError('Unauthorized access');
    expect(error).toBeInstanceOf(AuthError);
    expect(error).toBeInstanceOf(SievrantError);
    expect(error.statusCode).toBe(401);
    expect(error.message).toBe('Unauthorized access');
  });

  it('CorpusNotFoundError stores corpus name and extends SievrantError', () => {
    const error = new CorpusNotFoundError('test-corpus');
    expect(error).toBeInstanceOf(CorpusNotFoundError);
    expect(error).toBeInstanceOf(SievrantError);
    expect(error.corpus).toBe('test-corpus');
    expect(error.message).toContain('test-corpus');
  });

  it('SievrantServerError stores status_code and optional error_code', () => {
    const error = new SievrantServerError('Internal Server Error', 500, 'ERR_INTERNAL');
    expect(error).toBeInstanceOf(SievrantServerError);
    expect(error).toBeInstanceOf(SievrantError);
    expect(error.statusCode).toBe(500);
    expect(error.errorCode).toBe('ERR_INTERNAL');
  });

  it('SievrantTimeoutError stores timeout duration', () => {
    const error = new SievrantTimeoutError(100);
    expect(error).toBeInstanceOf(SievrantTimeoutError);
    expect(error).toBeInstanceOf(SievrantError);
    expect(error.timeoutMs).toBe(100);
  });
});
