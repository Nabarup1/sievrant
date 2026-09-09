/**
 * Sievrant error types.
 */

/** Base class for all Sievrant SDK errors. */
export class SievrantError extends Error {
  public readonly statusCode: number | undefined;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = 'SievrantError';
    this.statusCode = statusCode;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** Raised when the API key is missing or invalid (HTTP 401). */
export class AuthError extends SievrantError {
  constructor(message = 'Invalid or missing API key') {
    super(message, 401);
    this.name = 'AuthError';
  }
}

/** Raised when the requested corpus does not exist (HTTP 404). */
export class CorpusNotFoundError extends SievrantError {
  public readonly corpus: string;

  constructor(corpus: string) {
    super(`Corpus '${corpus}' not found`, 404);
    this.name = 'CorpusNotFoundError';
    this.corpus = corpus;
  }
}

/** Raised when the Sievrant server returns a 5xx response. */
export class SievrantServerError extends SievrantError {
  public readonly errorCode: string | undefined;

  constructor(message: string, statusCode: number, errorCode?: string) {
    super(message, statusCode);
    this.name = 'SievrantServerError';
    this.errorCode = errorCode;
  }
}

/** Raised when a request to the Sievrant server times out. */
export class SievrantTimeoutError extends SievrantError {
  public readonly timeoutMs: number;

  constructor(timeoutMs: number) {
    super(`Request timed out after ${timeoutMs}ms`);
    this.name = 'SievrantTimeoutError';
    this.timeoutMs = timeoutMs;
  }
}

/** Raised when the rate limit is exceeded (HTTP 429). */
export class RateLimitError extends SievrantError {
  public readonly retryAfter: number | undefined;

  constructor(retryAfter?: number) {
    const msg = retryAfter
      ? `Rate limit exceeded. Retry after ${retryAfter}s.`
      : 'Rate limit exceeded.';
    super(msg, 429);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}
