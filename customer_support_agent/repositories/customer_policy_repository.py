"""Customer policy repository.

get_active_policy_for_customer is the method the agent's
customer_history_lookup tool should call *before* the agent reasons about
coverage clauses at all -- if the policy isn't valid, the answer doesn't
need retrieval, it needs a "your policy expired/lapsed" response.
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import select

from customer_support_agent.models import CustomerPolicy, PolicyStatus
from customer_support_agent.repositories.base import BaseRepository


class CustomerPolicyRepository(BaseRepository[CustomerPolicy]):
    model = CustomerPolicy

    def create(
        self,
        customer_id: int,
        policy_document_id: int,
        policy_number: str,
        vehicle_reg_no: str,
        start_date: date,
        expiry_date: date,
        premium_amount: Decimal | str,
        status: PolicyStatus = PolicyStatus.ACTIVE,
    ) -> CustomerPolicy:
        return self.add(
            CustomerPolicy(
                customer_id=customer_id,
                policy_document_id=policy_document_id,
                policy_number=policy_number,
                vehicle_reg_no=vehicle_reg_no,
                start_date=start_date,
                expiry_date=expiry_date,
                premium_amount=premium_amount,
                status=status,
            )
        )

    def get_by_policy_number(self, policy_number: str) -> CustomerPolicy | None:
        stmt = select(CustomerPolicy).where(CustomerPolicy.policy_number == policy_number)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_for_customer(self, customer_id: int) -> list[CustomerPolicy]:
        stmt = (
            select(CustomerPolicy)
            .where(CustomerPolicy.customer_id == customer_id)
            .order_by(CustomerPolicy.start_date.desc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_active_policy_for_customer(
        self, customer_id: int, as_of: date | None = None
    ) -> CustomerPolicy | None:
        """Return the customer's currently-valid policy, if any.

        Deliberately does the date/status check in Python via
        CustomerPolicy.is_valid_on() rather than pure SQL, so the single
        source of truth for "what counts as valid" lives on the model,
        not duplicated across every caller.
        """
        as_of = as_of or date.today()
        candidates = self.list_for_customer(customer_id)
        for policy in candidates:
            if policy.is_valid_on(as_of):
                return policy
        return None
