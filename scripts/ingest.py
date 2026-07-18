"""Ingest every .md file under knowledge_base/ into Pinecone.

Usage:
    uv run python scripts/ingest.py

Safe to re-run any time: new files are added, changed files have their old
clauses cleanly replaced (see integrations/rag.py for how stale vectors are
tracked and deleted before re-upserting).

Reads OPENAI_API_KEY/OPENAI_BASE_URL, PINECONE_API_KEY, and DATABASE_URL
from .env via core.settings.
"""

from customer_support_agent.core import get_session, settings
from customer_support_agent.integrations.rag import ingest_all


def main() -> None:
    print(f"Target database: {settings.database_url}")
    print(f"Pinecone policies index: {settings.pinecone_policies_index}")
    print(f"Embedding model: {settings.embedding_model}\n")

    with get_session() as session:
        results = ingest_all(session)

    print(f"Ingested {len(results)} document(s):\n")
    print(f"{'Product':<22} {'Version':<10} {'Chunks':<8} {'Stale deleted':<14} {'Source'}")
    print("-" * 90)
    for r in results:
        print(
            f"{r.product_type:<22} {r.version:<10} {r.chunks_ingested:<8} "
            f"{r.deleted_stale_count:<14} {r.source_file}"
        )


if __name__ == "__main__":
    main()
