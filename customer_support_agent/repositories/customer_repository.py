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
