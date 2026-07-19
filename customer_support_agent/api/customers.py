"""Customer lookup -- used by the UI's login screen: customers type their
own email address (get_customer_by_email) rather than picking their name
from a dropdown, plus a single lookup (get_customer) kept for anything that
just needs to validate/display one customer by ID."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from customer_support_agent.api.deps import get_db
from customer_support_agent.repositories import CustomerRepository
from customer_support_agent.schemas.api import CustomerOut

router = APIRouter(prefix="/customers", tags=["customers"])


@router.get("", response_model=list[CustomerOut])
def list_customers(db: Session = Depends(get_db)) -> list[CustomerOut]:
    customers = CustomerRepository(db).list(limit=200)
    return [
        CustomerOut(id=c.id, name=c.name) for c in sorted(customers, key=lambda c: c.name)
    ]


@router.get("/lookup", response_model=CustomerOut)
def get_customer_by_email(email: str, db: Session = Depends(get_db)) -> CustomerOut:
    """Resolves the email a customer types on the login screen to an
    internal customer_id -- customer_id itself is never shown or typed by
    the user. Must be declared before GET /{customer_id} below, or the
    literal path "lookup" would be swallowed by that route's {customer_id}
    path param instead of reaching this one."""
    customer = CustomerRepository(db).get_by_email(email)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"No customer found for email {email!r}")
    return CustomerOut(id=customer.id, name=customer.name)


@router.get("/{customer_id}", response_model=CustomerOut)
def get_customer(customer_id: int, db: Session = Depends(get_db)) -> CustomerOut:
    customer = CustomerRepository(db).get(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return CustomerOut(id=customer.id, name=customer.name)
