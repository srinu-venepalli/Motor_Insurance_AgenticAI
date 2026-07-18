"""Tests for policy_parser.py -- pure parsing, no network involved."""

import pytest

from customer_support_agent.integrations.policy_parser import (
    FrontmatterError,
    parse_clauses,
    parse_frontmatter,
)

SAMPLE_DOC = """---
product_type: motor_test
version: v1.0
title: Test Policy
---

# Test Policy

## Section 1: Cover

### Clause AA-1.0: First Clause
This is the first clause body.

#### Clause AA-1.1: Sub Clause
This is a sub-clause body, still under Section 1.

## Section 2: Claims

### Clause BB-1.0: Second Clause
This is the second clause body, under Section 2.
"""


def test_parse_frontmatter_extracts_metadata_and_body():
    meta, body = parse_frontmatter(SAMPLE_DOC)
    assert meta.product_type == "motor_test"
    assert meta.version == "v1.0"
    assert meta.title == "Test Policy"
    assert "# Test Policy" in body
    assert "---" not in body.split("\n")[0]


def test_parse_frontmatter_missing_block_raises():
    with pytest.raises(FrontmatterError):
        parse_frontmatter("# No frontmatter here\nJust content.")


def test_parse_frontmatter_missing_required_key_raises():
    bad_doc = "---\ntitle: Missing product type and version\n---\nBody"
    with pytest.raises(FrontmatterError):
        parse_frontmatter(bad_doc)


def test_parse_clauses_splits_by_clause_heading():
    _, body = parse_frontmatter(SAMPLE_DOC)
    chunks = parse_clauses(body)

    assert [c.clause_id for c in chunks] == ["AA-1.0", "AA-1.1", "BB-1.0"]


def test_parse_clauses_attaches_correct_section_title():
    _, body = parse_frontmatter(SAMPLE_DOC)
    chunks = parse_clauses(body)
    by_id = {c.clause_id: c for c in chunks}

    assert by_id["AA-1.0"].section_title == "Cover"
    assert by_id["AA-1.1"].section_title == "Cover"  # still under Section 1
    assert by_id["BB-1.0"].section_title == "Claims"


def test_parse_clauses_text_does_not_bleed_into_next_clause():
    _, body = parse_frontmatter(SAMPLE_DOC)
    chunks = parse_clauses(body)
    by_id = {c.clause_id: c for c in chunks}

    assert "first clause body" in by_id["AA-1.0"].text
    assert "Sub Clause" not in by_id["AA-1.0"].text
    assert "sub-clause body" in by_id["AA-1.1"].text


def test_embedding_input_includes_section_and_clause_context():
    _, body = parse_frontmatter(SAMPLE_DOC)
    chunks = parse_clauses(body)
    embedding_text = chunks[0].embedding_input()

    assert "Cover" in embedding_text
    assert "AA-1.0" in embedding_text
    assert "First Clause" in embedding_text
    assert "first clause body" in embedding_text


def test_split_oversized_chunk_leaves_short_chunk_unchanged():
    from customer_support_agent.integrations.policy_parser import ClauseChunk, split_oversized_chunk

    chunk = ClauseChunk(
        clause_id="X-1.0", clause_title="Short", section_title="Sec", text="Short body."
    )
    result = split_oversized_chunk(chunk, max_chars=1200)

    assert result == [chunk]


def test_split_oversized_chunk_splits_long_text_with_suffixed_ids():
    from customer_support_agent.integrations.policy_parser import ClauseChunk, split_oversized_chunk

    long_text = "This is a sentence about coverage terms. " * 100  # ~4200 chars
    chunk = ClauseChunk(
        clause_id="X-9.0", clause_title="Long Clause", section_title="Sec", text=long_text
    )
    result = split_oversized_chunk(chunk, max_chars=1200, overlap=100)

    assert len(result) > 1
    ids = [c.clause_id for c in result]
    assert ids == [f"X-9.0-p{i + 1}" for i in range(len(result))]
    for sub_chunk in result:
        assert len(sub_chunk.text) <= 1200
        assert sub_chunk.section_title == "Sec"
        assert sub_chunk.clause_title == "Long Clause"
