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
from sievrant.graphrag import EntityExtractor
from sievrant.dspy_optimizer import PromptOptimizer
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
