"""Integrations package -- external service adapters (RAG/Pinecone, and
later memory + tool implementations for the LangGraph agent core)."""

from customer_support_agent.integrations.policy_parser import (
    ClauseChunk,
    DocumentMetadata,
    FrontmatterError,
    parse_clauses,
    parse_frontmatter,
    split_oversized_chunk,
)
from customer_support_agent.integrations.rag import (
    IngestResult,
    default_embed_fn,
    get_policies_index,
    ingest_all,
    ingest_file,
)

__all__ = [
    "ClauseChunk",
    "DocumentMetadata",
    "FrontmatterError",
    "parse_clauses",
    "parse_frontmatter",
    "split_oversized_chunk",
    "IngestResult",
    "default_embed_fn",
    "get_policies_index",
    "ingest_all",
    "ingest_file",
]
