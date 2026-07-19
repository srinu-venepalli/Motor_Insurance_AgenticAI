"""Customer repository."""

from sqlalchemy import select

from customer_support_agent.models import Customer
from customer_support_agent.repositories.base import BaseRepository


class CustomerRepository(BaseRepository[Customer]):
    model = Customer

    def create(self, name: str, contact_no: str, contact_email: str | None = None) -> Customer:
        return self.add(Customer(name=name, contact_no=contact_no, contact_email=contact_email))

    def get_by_contact_no(self, contact_no: str) -> Customer | None:
        """Look up a customer by phone -- useful when a ticket comes in
        without a customer_id yet and needs to be matched to an existing
        record."""
        stmt = select(Customer).where(Customer.contact_no == contact_no)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_many(self, ids: list[int]) -> dict[int, Customer]:
        """Batch fetch -- one query for N ids, instead of N separate
        .get() calls. Used by list_tickets() to avoid an N+1 query per
        ticket in the queue/history views."""
        if not ids:
            return {}
        stmt = select(Customer).where(Customer.id.in_(ids))
        return {c.id: c for c in self.session.execute(stmt).scalars().all()}
