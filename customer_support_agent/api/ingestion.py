"""Ingestion endpoint -- lets you trigger a re-scan of knowledge_base/
without SSH-ing in to run scripts/ingest.py manually. Same underlying
ingest_all() the CLI script calls, so behaviour (including the
replace-stale-vectors logic) is identical either way.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from customer_support_agent.api.deps import get_db
from customer_support_agent.core import get_logger
from customer_support_agent.integrations.rag import ingest_all

router = APIRouter(prefix="/ingest", tags=["ingestion"])
logger = get_logger(__name__)


@router.post("")
def trigger_ingestion(db: Session = Depends(get_db)) -> dict:
    results = ingest_all(db)
    return {
        "documents_ingested": len(results),
        "results": [
            {
                "source_file": r.source_file,
                "product_type": r.product_type,
                "version": r.version,
                "chunks_ingested": r.chunks_ingested,
                "deleted_stale_count": r.deleted_stale_count,
            }
            for r in results
        ],
    }
