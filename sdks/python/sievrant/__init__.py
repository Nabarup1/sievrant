"""
Sievrant Python SDK

The statistically audited retrieval decision engine for enterprise AI.
"""

from sievrant.client import SievrantClient
from sievrant.models import (
    EvidencePacket,
    EvidenceChunk,
    AuditBlock,
    RetrievalStagesMs,
    Decision,
    Route,
    RetrieverType,
    HealthResponse,
    IngestResponse,
    CorpusMetadata,
)
from sievrant.exceptions import (
    SievrantError,
    AuthError,
    CorpusNotFoundError,
    SievrantServerError,
    SievrantTimeoutError,
    SievrantAbstainedError,
)

__all__ = [
    "SievrantClient",
    "EvidencePacket",
    "EvidenceChunk",
    "AuditBlock",
    "RetrievalStagesMs",
    "Decision",
    "Route",
    "RetrieverType",
    "HealthResponse",
    "IngestResponse",
    "CorpusMetadata",
    "SievrantError",
    "AuthError",
    "CorpusNotFoundError",
    "SievrantServerError",
    "SievrantTimeoutError",
    "SievrantAbstainedError",
]

__version__ = "0.1.0"
