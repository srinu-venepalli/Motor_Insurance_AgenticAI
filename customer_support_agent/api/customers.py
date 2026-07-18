"""Customer lookup -- used by the UI's login screen: a dropdown of names
(list_customers) rather than asking a customer to know their own internal
database ID, plus a single lookup (get_customer) kept for anything that
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


@router.get("/{customer_id}", response_model=CustomerOut)
def get_customer(customer_id: int, db: Session = Depends(get_db)) -> CustomerOut:
    customer = CustomerRepository(db).get(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return CustomerOut(id=customer.id, name=customer.name)
