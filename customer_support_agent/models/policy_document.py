"""Policy document table.

One row per policy PDF/version that gets chunked and embedded into the
Pinecone `policies` index (see integrations/rag.py for the ingestion side).
Distinct from CustomerPolicy, which is a specific customer's coverage
instance tied back to one of these documents.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from customer_support_agent.models.base import Base, IntPKMixin

if TYPE_CHECKING:
    from customer_support_agent.models.customer_policy import CustomerPolicy


class PolicyDocument(Base, IntPKMixin):
    __tablename__ = "policy_documents"

    product_type: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    source_file: Mapped[str] = mapped_column(String(500), nullable=False)
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # IDs of the vectors currently upserted into Pinecone for this document.
    # Tracked here (not derivable from Pinecone alone, since serverless
    # indexes can only delete by ID, not by metadata filter) so re-ingestion
    # can cleanly delete the old set before upserting the new one -- this is
    # what makes "existing ones should be overridden" actually correct even
    # when clauses are added/removed/renamed between versions.
    last_ingested_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)

    customer_policies: Mapped[list["CustomerPolicy"]] = relationship(
        back_populates="policy_document"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"PolicyDocument(id={self.id}, product_type={self.product_type!r}, version={self.version!r})"
