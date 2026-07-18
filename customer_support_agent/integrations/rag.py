"""RAG ingestion: knowledge_base/*.md -> chunked, embedded, upserted into the
Pinecone `policies` index, with the Postgres policy_documents table tracking
what's currently indexed so re-ingestion cleanly replaces rather than
accumulates stale vectors.

Embedding and Pinecone clients are injectable (embed_fn / index params)
specifically so this module is unit-testable without real API keys -- see
tests/test_rag_ingestion.py for the fake-client version, and
scripts/ingest.py / the /ingest API endpoint for the real-client version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from sqlalchemy.orm import Session

from customer_support_agent.core import get_logger, log_transaction, settings
from customer_support_agent.integrations.policy_parser import (
    ClauseChunk,
    DocumentMetadata,
    parse_clauses,
    parse_frontmatter,
    split_oversized_chunk,
)
from customer_support_agent.repositories import PolicyDocumentRepository

logger = get_logger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]


class VectorIndex(Protocol):
    """The subset of the Pinecone Index interface we actually use -- typed
    as a Protocol so tests can pass in a plain fake object instead of a real
    Pinecone client."""

    def upsert(self, vectors: list[dict]) -> object: ...
    def delete(self, ids: list[str]) -> object: ...


@dataclass
class IngestResult:
    source_file: str
    product_type: str
    version: str
    chunks_ingested: int
    chunk_ids: list[str] = field(default_factory=list)
    deleted_stale_count: int = 0


# --- Real clients (lazily constructed; not called at import time so tests
# never need real API keys just to import this module) -------------------


def default_embed_fn(texts: list[str]) -> list[list[float]]:
    """Embeddings via LangChain's OpenAIEmbeddings (not a raw OpenAI client
    call) -- this is the project's declared Track A framework, so the RAG
    pipeline should visibly go through it rather than around it."""
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    return embeddings.embed_documents(texts)


_pinecone_client: Pinecone | None = None


def get_policies_index() -> VectorIndex:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pinecone_client.Index(settings.pinecone_policies_index)


# --- Core ingestion logic -------------------------------------------------


def _vector_id(product_type: str, version: str, clause_id: str) -> str:
    """Deterministic ID so re-ingesting the same clause upserts (overwrites)
    rather than creating a duplicate vector."""
    return f"{product_type}::{version}::{clause_id}"


def _build_vector(
    meta: DocumentMetadata,
    chunk: ClauseChunk,
    embedding: list[float],
    policy_document_id: int,
    source_file: str,
    ingested_at: datetime,
) -> dict:
    return {
        "id": _vector_id(meta.product_type, meta.version, chunk.clause_id),
        "values": embedding,
        "metadata": {
            "policy_doc_name": meta.title,
            "policy_version": meta.version,
            "policy_created": ingested_at.isoformat(),
            "product_type": meta.product_type,
            "policy_document_id": policy_document_id,
            "clause_id": chunk.clause_id,
            "clause_title": chunk.clause_title,
            "section": chunk.section_title,
            "source_file": source_file,
            "text": chunk.text,
        },
    }


def ingest_file(
    session: Session,
    file_path: Path,
    embed_fn: EmbedFn | None = None,
    index: VectorIndex | None = None,
) -> IngestResult:
    """Ingest a single knowledge_base Markdown file: parse -> embed -> upsert,
    deleting any previously-ingested vectors for this same document first so
    removed/renamed clauses don't linger as stale vectors.
    """
    embed_fn = embed_fn or default_embed_fn
    index = index or get_policies_index()

    raw_text = file_path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw_text)
    clauses = parse_clauses(body)
    # Primary split is clause-boundary (above); this flat-map only touches
    # clauses that exceed the size threshold -- most pass through as a
    # single-element list, unchanged.
    chunks = [sub for clause in clauses for sub in split_oversized_chunk(clause)]

    with log_transaction(
        "ingest_policy_document",
        source_file=str(file_path),
        product_type=meta.product_type,
        version=meta.version,
        chunk_count=len(chunks),
    ):
        doc_repo = PolicyDocumentRepository(session)
        doc = doc_repo.get_by_product_and_version(meta.product_type, meta.version)
        if doc is None:
            doc = doc_repo.create(
                product_type=meta.product_type,
                version=meta.version,
                source_file=str(file_path),
            )
            session.flush()

        stale_ids = list(doc.last_ingested_chunk_ids or [])
        if stale_ids:
            index.delete(ids=stale_ids)
            logger.info(
                "Deleted %d stale vector(s) for %s %s before re-ingesting",
                len(stale_ids),
                meta.product_type,
                meta.version,
            )

        ingested_at = datetime.now(timezone.utc)
        embeddings = embed_fn([c.embedding_input() for c in chunks]) if chunks else []

        vectors = [
            _build_vector(meta, chunk, embedding, doc.id, str(file_path), ingested_at)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        if vectors:
            index.upsert(vectors=vectors)

        new_chunk_ids = [v["id"] for v in vectors]
        doc.source_file = str(file_path)
        doc.ingested_at = ingested_at
        doc.last_ingested_chunk_ids = new_chunk_ids
        session.flush()

    return IngestResult(
        source_file=str(file_path),
        product_type=meta.product_type,
        version=meta.version,
        chunks_ingested=len(vectors) if chunks else 0,
        chunk_ids=new_chunk_ids,
        deleted_stale_count=len(stale_ids),
    )


def ingest_all(
    session: Session,
    knowledge_base_dir: Path | str = "knowledge_base",
    embed_fn: EmbedFn | None = None,
    index: VectorIndex | None = None,
) -> list[IngestResult]:
    """Ingest every .md file under knowledge_base_dir. Safe to re-run any
    time (adds new files, replaces changed ones) -- see ingest_file() for
    the per-file replace semantics."""
    kb_dir = Path(knowledge_base_dir)
    md_files = sorted(kb_dir.glob("*.md"))

    results = []
    with log_transaction("ingest_all", knowledge_base_dir=str(kb_dir), file_count=len(md_files)):
        for file_path in md_files:
            result = ingest_file(session, file_path, embed_fn=embed_fn, index=index)
            results.append(result)
    return results
