"""Typed application settings, loaded from environment variables / .env file.

Import `settings` from this module anywhere in the app instead of calling
os.environ directly, so config stays centralized and testable.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Motor Insurance Customer Support Agent"
    environment: str = Field(default="local")  # local | docker | production
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL the Streamlit UI uses to call the FastAPI backend.",
    )
    demo_shared_password: str = Field(
        default="password1",
        description="Hardcoded shared password for the demo UI login. NOT for production use.",
    )

    # --- Database (Postgres in docker-compose; sqlite fallback for quick local dev) ---
    database_url: str = Field(
        default="sqlite:///./local.db",
        description="SQLAlchemy connection string. Example (docker-compose postgres): "
        "postgresql+psycopg2://agent:localdev@localhost:5432/insurance_support",
    )

    # --- OpenAI (IITM-provided key + custom base url) ---
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)
    chat_model: str = Field(default="gpt-4.1")

    # --- Pinecone ---
    pinecone_api_key: str = Field(default="")
    pinecone_policies_index: str = Field(default="policies")
    pinecone_customer_memory_index: str = Field(default="customer-memory")

    # --- Observability ---
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="motor-insurance-support-agent")
    langsmith_tracing: bool = Field(default=False)

    # --- Application logging (separate from LangSmith tracing) ---
    log_dir: str = Field(default="logs")
    log_file: str = Field(default="app.log")
    log_level: str = Field(default="INFO")
    log_max_bytes: int = Field(default=5_000_000)  # ~5MB per file before rotating
    log_backup_count: int = Field(default=5)

    # --- Escalation business rules (see graph/nodes.py _evaluate_escalation) ---
    high_value_claim_threshold: float = Field(
        default=100_000,
        description="Claims above this amount (INR) always escalate to a human, regardless of confidence.",
    )
    max_tickets_per_30_days: int = Field(
        default=2,
        description="More than this many tickets from the same customer in the rolling window escalates.",
    )
    repeat_ticket_window_days: int = Field(
        default=30,
        description="Rolling window (days) used to count a customer's recent tickets.",
    )
    supervisor_agent_name: str = Field(
        default="Arjun Mehta",
        description="Name of the seeded agent (see services/seed_data.py) assigned to HIGH priority escalations.",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance -- import this, not Settings() directly."""
    return Settings()


settings = get_settings()
