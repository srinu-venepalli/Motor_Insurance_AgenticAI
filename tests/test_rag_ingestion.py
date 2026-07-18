"""Tests for rag.py -- the ingestion orchestration.

Uses fake embed_fn / index objects throughout so these tests need no real
OpenAI or Pinecone credentials. See test_ingest_real_knowledge_base_files
for a sanity check against the actual project knowledge_base/ content
(still with fakes -- it only proves the real files parse and chunk without
errors, not that real embeddings/upserts work).
"""

from pathlib import Path

from customer_support_agent.integrations.rag import ingest_all, ingest_file
from customer_support_agent.repositories import PolicyDocumentRepository

SAMPLE_V1 = """---
product_type: test_product
version: v1.0
title: Test Product Policy
---

## Section 1: Cover

### Clause X-1.0: First Clause
Body of the first clause.

### Clause X-2.0: Second Clause
Body of the second clause.
"""

# Same product/version, but X-2.0 removed and a new X-3.0 added -- simulates
# editing a policy file in place.
SAMPLE_V1_EDITED = """---
product_type: test_product
version: v1.0
title: Test Product Policy
---

## Section 1: Cover

### Clause X-1.0: First Clause
Body of the first clause, slightly reworded.

### Clause X-3.0: Third Clause
Body of a brand new clause.
"""


class FakeIndex:
    def __init__(self):
        self.upsert_calls: list[list[dict]] = []
        self.delete_calls: list[list[str]] = []

    def upsert(self, vectors):
        self.upsert_calls.append(vectors)

    def delete(self, ids):
        self.delete_calls.append(list(ids))


def fake_embed_fn(texts: list[str]) -> list[list[float]]:
    # Deterministic, cheap fake embedding: length-based vector, no network.
    return [[float(len(t)), 0.0, 0.0] for t in texts]


def test_ingest_file_creates_policy_document_and_upserts_vectors(session, tmp_path):
    file_path = tmp_path / "test_product_v1.md"
    file_path.write_text(SAMPLE_V1)
    index = FakeIndex()

    result = ingest_file(session, file_path, embed_fn=fake_embed_fn, index=index)
    session.commit()

    assert result.chunks_ingested == 2
    assert result.deleted_stale_count == 0
    assert len(index.upsert_calls) == 1
    vectors = index.upsert_calls[0]
    assert len(vectors) == 2

    ids = {v["id"] for v in vectors}
    assert ids == {"test_product::v1.0::X-1.0", "test_product::v1.0::X-2.0"}

    sample_meta = vectors[0]["metadata"]
    assert sample_meta["policy_doc_name"] == "Test Product Policy"
    assert sample_meta["policy_version"] == "v1.0"
    assert "policy_created" in sample_meta
    assert sample_meta["product_type"] == "test_product"
    assert "text" in sample_meta

    doc_repo = PolicyDocumentRepository(session)
    doc = doc_repo.get_by_product_and_version("test_product", "v1.0")
    assert doc is not None
    assert doc.ingested_at is not None
    assert set(doc.last_ingested_chunk_ids) == ids


def test_reingest_deletes_stale_vectors_and_replaces_them(session, tmp_path):
    file_path = tmp_path / "test_product_v1.md"
    index = FakeIndex()

    file_path.write_text(SAMPLE_V1)
    first_result = ingest_file(session, file_path, embed_fn=fake_embed_fn, index=index)
    session.commit()

    # Now "edit" the file: X-2.0 is gone, X-3.0 is new.
    file_path.write_text(SAMPLE_V1_EDITED)
    second_result = ingest_file(session, file_path, embed_fn=fake_embed_fn, index=index)
    session.commit()

    # The delete call before the second upsert must contain exactly the
    # first ingestion's ids, so the removed clause (X-2.0) doesn't linger.
    assert index.delete_calls == [sorted(first_result.chunk_ids)] or set(
        index.delete_calls[0]
    ) == set(first_result.chunk_ids)
    assert second_result.deleted_stale_count == 2

    doc_repo = PolicyDocumentRepository(session)
    doc = doc_repo.get_by_product_and_version("test_product", "v1.0")
    assert set(doc.last_ingested_chunk_ids) == {
        "test_product::v1.0::X-1.0",
        "test_product::v1.0::X-3.0",
    }
    # Only one policy_documents row -- re-ingestion updates in place, doesn't duplicate.
    assert doc_repo.get_by_product_and_version("test_product", "v1.0").id == (
        doc.id
    )


def test_ingest_all_processes_every_md_file_in_directory(session, tmp_path):
    (tmp_path / "doc_a.md").write_text(SAMPLE_V1)
    (tmp_path / "doc_b.md").write_text(
        SAMPLE_V1.replace("test_product", "other_product").replace("v1.0", "v2.0")
    )
    index = FakeIndex()

    results = ingest_all(session, knowledge_base_dir=tmp_path, embed_fn=fake_embed_fn, index=index)
    session.commit()

    assert len(results) == 2
    product_types = {r.product_type for r in results}
    assert product_types == {"test_product", "other_product"}
    assert all(r.chunks_ingested == 2 for r in results)


def test_ingest_real_knowledge_base_files_parse_and_chunk_without_error(session):
    """Sanity check against the project's actual knowledge_base/ files --
    still uses fakes for embed/index, so no real API calls happen."""
    kb_dir = Path("knowledge_base")
    if not kb_dir.exists():
        return  # skip gracefully if run from an unexpected cwd

    index = FakeIndex()
    results = ingest_all(session, knowledge_base_dir=kb_dir, embed_fn=fake_embed_fn, index=index)
    session.commit()

    assert len(results) == 2
    total_chunks = sum(r.chunks_ingested for r in results)
    assert total_chunks > 30  # 24 + 12 clauses at time of writing


SAMPLE_WITH_OVERSIZED_CLAUSE = """---
product_type: test_product
version: v1.0
title: Test Product Policy
---

## Section 1: Cover

### Clause X-1.0: Normal Clause
A short, normal-sized clause body.

### Clause X-2.0: Oversized Clause
""" + ("This clause has unusually dense legal text repeated many times. " * 60)


def test_ingest_file_splits_oversized_clause_into_multiple_vectors(session, tmp_path):
    file_path = tmp_path / "oversized_test.md"
    file_path.write_text(SAMPLE_WITH_OVERSIZED_CLAUSE)
    index = FakeIndex()

    result = ingest_file(session, file_path, embed_fn=fake_embed_fn, index=index)
    session.commit()

    # X-1.0 stays as one vector; X-2.0 (oversized) becomes multiple, suffixed.
    assert result.chunks_ingested > 2
    ids = result.chunk_ids
    assert "test_product::v1.0::X-1.0" in ids
    assert any("X-2.0-p1" in vid for vid in ids)
    assert any("X-2.0-p2" in vid for vid in ids)
