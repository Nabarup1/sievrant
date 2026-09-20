"""
Sievrant Python SDK

The statistically audited retrieval decision engine for enterprise AI.
"""

from sievrant.client import SievrantClient
from sievrant.dspy_optimizer import PromptOptimizer
from sievrant.exceptions import (
    AuthError,
    CorpusNotFoundError,
    SievrantAbstainedError,
    SievrantError,
    SievrantServerError,
    SievrantTimeoutError,
)
from sievrant.graphrag import EntityExtractor
from sievrant.models import (
    AuditBlock,
    CorpusMetadata,
    Decision,
    EvidenceChunk,
    EvidencePacket,
    HealthResponse,
    IngestResponse,
    RetrievalStagesMs,
    RetrieverType,
    Route,
)
from sievrant.raptor import TreeSummarizer

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
    "EntityExtractor",
    "PromptOptimizer",
    "TreeSummarizer",
]

__version__ = "0.1.0"
