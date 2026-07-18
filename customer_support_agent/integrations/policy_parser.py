"""Parsing helpers for policy documents: YAML frontmatter + clause-level
chunking. Pure functions, no network calls -- kept separate from the
embedding/Pinecone client code in rag.py so they're trivially unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)

# Matches "### Clause OD-4.2: Parked Vehicle -- Unknown Cause" and the
# "####" sub-clause variant. Captures the clause id and its title.
_CLAUSE_HEADING_RE = re.compile(
    r"^#{2,4}\s*Clause\s+([A-Za-z0-9\-\.]+):\s*(.+?)\s*$", re.MULTILINE
)
_SECTION_HEADING_RE = re.compile(r"^##\s*Section\s+\d+:\s*(.+?)\s*$", re.MULTILINE)


class FrontmatterError(ValueError):
    """Raised when a knowledge_base file is missing required frontmatter."""


@dataclass(frozen=True)
class DocumentMetadata:
    product_type: str
    version: str
    title: str


def parse_frontmatter(raw_text: str) -> tuple[DocumentMetadata, str]:
    """Split a Markdown file into (metadata, body).

    Expects a YAML frontmatter block at the very top:

        ---
        product_type: motor_comprehensive
        version: v3.2
        title: Motor Comprehensive Insurance Policy
        ---
        # rest of the document...

    Raises FrontmatterError if the block is missing or required keys
    (product_type, version) are absent -- deliberately strict, since a
    silently-skipped file would be a much harder bug to notice than an
    ingestion error.
    """
    match = _FRONTMATTER_RE.match(raw_text)
    if not match:
        raise FrontmatterError(
            "No YAML frontmatter block found at the top of the file. "
            "Expected a '---'-delimited block with at least product_type and version."
        )
    raw_meta, body = match.group(1), match.group(2)
    data = yaml.safe_load(raw_meta) or {}

    missing = [key for key in ("product_type", "version") if key not in data]
    if missing:
        raise FrontmatterError(f"Frontmatter missing required key(s): {missing}")

    return (
        DocumentMetadata(
            product_type=str(data["product_type"]),
            version=str(data["version"]),
            title=str(data.get("title", data["product_type"])),
        ),
        body,
    )


@dataclass(frozen=True)
class ClauseChunk:
    clause_id: str
    clause_title: str
    section_title: str
    text: str

    def embedding_input(self) -> str:
        """The actual string to embed -- includes section + clause title as
        context so short clause bodies aren't embedded without any framing."""
        header = f"{self.section_title} — Clause {self.clause_id}: {self.clause_title}"
        return f"{header}\n\n{self.text.strip()}"


def parse_clauses(body: str) -> list[ClauseChunk]:
    """Split a policy document body into one chunk per clause.

    Chunk boundaries are '###'/'####' headings matching "Clause <ID>: <title>".
    Each chunk's text runs from just after its heading to the start of the
    next clause heading (or end of file). The nearest preceding
    '## Section N: <title>' heading is attached as section_title context.
    """
    section_matches = [(m.start(), m.group(1)) for m in _SECTION_HEADING_RE.finditer(body)]
    clause_matches = [
        (m.start(), m.end(), m.group(1), m.group(2))
        for m in _CLAUSE_HEADING_RE.finditer(body)
    ]

    def section_title_before(pos: int) -> str:
        title = ""
        for start, section_title in section_matches:
            if start < pos:
                title = section_title
            else:
                break
        return title

    chunks: list[ClauseChunk] = []
    for i, (start, header_end, clause_id, clause_title) in enumerate(clause_matches):
        next_start = clause_matches[i + 1][0] if i + 1 < len(clause_matches) else len(body)
        text = body[header_end:next_start].strip()
        chunks.append(
            ClauseChunk(
                clause_id=clause_id,
                clause_title=clause_title,
                section_title=section_title_before(start),
                text=text,
            )
        )
    return chunks


# Clause-boundary splitting (above) is the primary chunking strategy, since
# it gives clean, citation-ready units for a well-structured policy doc. But
# nothing guarantees a real insurer's clause stays short -- if one runs long
# (e.g. dense legal fine print), embedding it as one oversized chunk hurts
# retrieval precision and risks exceeding the embedding model's practical
# input size. RecursiveCharacterTextSplitter here is a safety net, not the
# primary strategy: most clauses pass through untouched (single-element
# list), only oversized ones get sub-split.
_DEFAULT_MAX_CHARS = 1200
_DEFAULT_OVERLAP = 150


def split_oversized_chunk(
    chunk: ClauseChunk,
    max_chars: int = _DEFAULT_MAX_CHARS,
    overlap: int = _DEFAULT_OVERLAP,
) -> list[ClauseChunk]:
    """Return [chunk] unchanged if it's within max_chars. Otherwise split its
    text with LangChain's RecursiveCharacterTextSplitter and return multiple
    ClauseChunks, each with a distinguishing clause_id suffix
    (e.g. "OD-4.2" -> "OD-4.2-p1", "OD-4.2-p2", ...) so each still gets a
    unique, traceable vector ID downstream.
    """
    if len(chunk.text) <= max_chars:
        return [chunk]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    parts = splitter.split_text(chunk.text)
    return [
        ClauseChunk(
            clause_id=f"{chunk.clause_id}-p{i + 1}",
            clause_title=chunk.clause_title,
            section_title=chunk.section_title,
            text=part,
        )
        for i, part in enumerate(parts)
    ]
