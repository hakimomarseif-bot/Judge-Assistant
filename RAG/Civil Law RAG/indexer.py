"""
indexer.py

Civil Law indexing pipeline.

Purpose:
---------
Responsible for one-time indexing of the Egyptian Civil Law
into the Qdrant vector database.

Pipeline:
---------
1. Load raw text file
2. Split using hierarchical splitter
3. Batch embed documents
4. Persist into Qdrant (server-side, automatic)

Important:
----------
The ``ensure_civil_law_indexed()`` helper is designed to be called at
application startup.  It checks whether the Qdrant collection already
contains data and only runs the full indexing pipeline when the
collection is empty -- making repeated calls a cheap no-op.

Why this exists:
----------------
Indexing is computationally expensive and must be isolated from
query-time logic.

Design Principle:
-----------------
Clear separation between:
- Offline indexing stage
- Online retrieval & reasoning stage
"""
import logging
import os

from langchain_community.document_loaders import TextLoader
 
from config.rag import DOCS_PATH, BATCH_SIZE
from splitter import split_egyptian_civil_law
from vectorstore import load_vectorstore, _get_qdrant_client
from config import cfg

logger = logging.getLogger("civil_law_rag.indexer")


def _collection_point_count() -> int:
    """Return the number of points currently stored in the Qdrant collection."""
    client = _get_qdrant_client()
    collection_name = cfg.qdrant.get("collection", "judicial_docs")
    try:
        info = client.get_collection(collection_name)
        return info.points_count or 0
    except Exception as exc:
        # Only treat "collection not found" as empty; re-raise other errors
        # so transient connectivity issues don't trigger duplicate indexing.
        _not_found = getattr(exc, "status_code", None) == 404 or "not found" in str(exc).lower()
        if _not_found:
            return 0
        raise


def index_civil_law():
    """Run the full civil-law indexing pipeline.

    Raises ``FileNotFoundError`` if the source text file is missing.
    Skips indexing when the Qdrant collection already has data.
    """
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"{DOCS_PATH} not found")

    db = load_vectorstore()

    if _collection_point_count() > 0:
        logger.info("Qdrant collection already populated. Skipping indexing.")
        return db

    logger.info("Indexing Egyptian Civil Law into Qdrant...")

    loader = TextLoader(DOCS_PATH, encoding="utf-8")
    document = loader.load()

    raw_text = document[0].page_content
    docs = split_egyptian_civil_law(raw_text)

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        db.add_documents(batch)
        logger.info("Indexed %d / %d documents", i + len(batch), len(docs))

    logger.info("Civil law indexing completed.")
    return db


def ensure_civil_law_indexed():
    """Ensure the civil law corpus is present in Qdrant.

    This is a fast no-op when data already exists (single API call to check
    point count).  When the collection is empty, the full indexing pipeline
    runs automatically.
    """
    if _collection_point_count() > 0:
        logger.debug("Civil law data already present in Qdrant -- skipping.")
        return

    logger.info("Civil law collection is empty -- running auto-indexing...")
    index_civil_law()
