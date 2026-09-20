"""
Exceptions raised by the Sievrant Python SDK.
All exceptions inherit from SievrantError for easy catch-all handling.
"""

from __future__ import annotations

from typing import Any


class SievrantError(Exception):
    """Base class for all Sievrant SDK errors."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthError(SievrantError):
    """Raised when the API key is missing or invalid (HTTP 401)."""

    def __init__(self, message: str = "Invalid or missing API key") -> None:
        super().__init__(message, status_code=401)


class CorpusNotFoundError(SievrantError):
    """Raised when the requested corpus does not exist (HTTP 404)."""

    def __init__(self, corpus: str) -> None:
        self.corpus = corpus
        super().__init__(f"Corpus '{corpus}' not found", status_code=404)


class SievrantServerError(SievrantError):
    """Raised when the Sievrant server returns a 5xx response."""

    def __init__(
        self,
        message: str,
        status_code: int,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.error_code = error_code


class SievrantTimeoutError(SievrantError):
    """Raised when a request to the Sievrant server times out."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Request timed out after {timeout_seconds}s")


class SievrantAbstainedError(SievrantError):
    """Raised when Sievrant returns Decision.ABSTAINED and raise_on_abstain=True.

    Contains the full query_id and audit block for logging purposes.
    """

    def __init__(self, query_id: str, audit: Any) -> None:
        self.query_id = query_id
        self.audit = audit
        super().__init__(
            f"Sievrant abstained on query {query_id}: insufficient evidence to guarantee coverage"
        )


class ValidationError(SievrantError):
    """Raised when the request payload fails server-side validation (HTTP 422)."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        super().__init__(message, status_code=422)


class RateLimitError(SievrantError):
    """Raised when the rate limit for the API key is exceeded (HTTP 429)."""

    def __init__(self, retry_after: int | None = None) -> None:
        self.retry_after = retry_after
        msg = "Rate limit exceeded"
        if retry_after is not None:
            msg += f". Retry after {retry_after} seconds."
        super().__init__(msg, status_code=429)
