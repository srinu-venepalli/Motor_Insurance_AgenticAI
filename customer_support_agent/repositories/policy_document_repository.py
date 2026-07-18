"""Policy document repository -- metadata for PDFs chunked into Pinecone."""

from datetime import datetime

from sqlalchemy import select

from customer_support_agent.models import PolicyDocument
from customer_support_agent.repositories.base import BaseRepository


class PolicyDocumentRepository(BaseRepository[PolicyDocument]):
    model = PolicyDocument

    def create(
        self,
        product_type: str,
        version: str,
        source_file: str,
        ingested_at: datetime | None = None,
    ) -> PolicyDocument:
        return self.add(
            PolicyDocument(
                product_type=product_type,
                version=version,
                source_file=source_file,
                ingested_at=ingested_at,
            )
        )

    def get_by_product_and_version(self, product_type: str, version: str) -> PolicyDocument | None:
        stmt = select(PolicyDocument).where(
            PolicyDocument.product_type == product_type,
            PolicyDocument.version == version,
        )
        return self.session.execute(stmt).scalar_one_or_none()
