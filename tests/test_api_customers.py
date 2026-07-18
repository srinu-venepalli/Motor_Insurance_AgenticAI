"""Tests for GET /customers/{id} -- used by the Streamlit login screen to
validate a customer ID and fetch their display name."""

from fastapi.testclient import TestClient

from customer_support_agent.api.deps import get_db
from customer_support_agent.repositories import CustomerRepository
from main import app


def _client(session):
    def override_get_db():
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_get_customer_success(session):
    customer = CustomerRepository(session).create(name="Rohan Sharma", contact_no="+91-9800000001")
    session.commit()

    client = _client(session)
    response = client.get(f"/customers/{customer.id}")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"id": customer.id, "name": "Rohan Sharma"}


def test_get_customer_not_found(session):
    client = _client(session)
    response = client.get("/customers/999999")
    app.dependency_overrides.clear()

    assert response.status_code == 404
