"""case_doc_rag.infrastructure -- Lazy singleton accessors for external connections.

Nothing runs at import time. This is the only file in the package that
imports from config/. All other files go through these accessors.

Design decision: get_llm(tier) does NOT forward **overrides to the config
factory because (a) no node in this pipeline needs overrides, (b) caching
by tier becomes ambiguous if overrides differ between calls, and (c) if
overrides are needed in the future the cache key must incorporate them.
"""

import logging
import threading
from typing import Any, Dict, Optional

from config import get_llm as _config_get_llm
from config import cfg

logger = logging.getLogger("case_doc_rag.infrastructure")

# ---------------------------------------------------------------------------
# Private singletons -- all start as None; nothing initializes at import time
# ---------------------------------------------------------------------------
_embedding_fn = None
_llm_cache: Dict[str, Any] = {}
_qdrant_client = None
_vectorstore = None
_mongo_collection = None

# A single reentrant lock guards all singleton accessors.  RLock is used
# (instead of Lock) because get_vectorstore() calls get_qdrant_client()
# and get_embedding_function() while holding the lock.  All accessors use
# double-checked locking: the fast path (already initialised) returns
# without acquiring the lock; the slow path acquires the lock and rechecks.
_singleton_lock = threading.RLock()


def get_embedding_function():
    """Return the shared HuggingFaceEmbeddings instance (lazy, thread-safe)."""
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn
    with _singleton_lock:
        if _embedding_fn is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            model_name = cfg.embedding.get("model", "BAAI/bge-m3")
            logger.info("Initializing HuggingFaceEmbeddings with model=%s", model_name)
            _embedding_fn = HuggingFaceEmbeddings(model_name=model_name)
    return _embedding_fn


def get_llm(tier: str):
    """Return a cached LangChain chat model for the requested tier (thread-safe).

    Calls config.get_llm(tier) on first access for each tier and caches
    the result. Does not forward **overrides -- see module docstring.
    """
    if tier in _llm_cache:
        return _llm_cache[tier]
    with _singleton_lock:
        if tier not in _llm_cache:
            logger.info("Initializing LLM for tier=%s", tier)
            _llm_cache[tier] = _config_get_llm(tier)
    return _llm_cache[tier]


def get_qdrant_client():
    """Return the shared QdrantClient instance (lazy, thread-safe)."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    with _singleton_lock:
        if _qdrant_client is None:
            from qdrant_client import QdrantClient

            host = cfg.qdrant.get("host", "localhost")
            port = cfg.qdrant.get("port", 6333)
            grpc_port = cfg.qdrant.get("grpc_port", 6334)
            prefer_grpc = cfg.qdrant.get("prefer_grpc", True)
            logger.info(
                "Initializing QdrantClient host=%s port=%s grpc_port=%s",
                host, port, grpc_port,
            )
            _qdrant_client = QdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
            )
    return _qdrant_client


def get_vectorstore():
    """Return the shared QdrantVectorStore instance (lazy, thread-safe)."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    with _singleton_lock:
        if _vectorstore is None:
            from langchain_qdrant import QdrantVectorStore

            collection_name = cfg.qdrant.get("collection", "judicial_docs")
            logger.info(
                "Initializing QdrantVectorStore collection=%s", collection_name
            )
            _vectorstore = QdrantVectorStore(
                client=get_qdrant_client(),
                collection_name=collection_name,
                embedding=get_embedding_function(),
            )
        return _vectorstore


def set_vectorstore(vectorstore) -> None:
    """Inject an externally-provided vector store (thread-safe).

    This is how the Supervisor injects its already-initialized Qdrant
    instance so that documents indexed at ingest time are visible
    during retrieval.
    """
    global _vectorstore
    with _singleton_lock:
        _vectorstore = vectorstore
        logger.info("Vectorstore replaced via set_vectorstore()")


def get_retriever(search_kwargs: Optional[dict] = None):
    """Build a fresh retriever from the shared vectorstore.

    Does NOT cache -- filters and thresholds vary per call so caching
    would return a retriever with wrong parameters. Used only for
    unfiltered fallback in retrieval_nodes.
    """
    kwargs = search_kwargs or {"k": 15}
    return get_vectorstore().as_retriever(
        search_type="mmr", search_kwargs=kwargs
    )


def get_mongo_collection():
    """Return the shared MongoDB collection (lazy, thread-safe)."""
    global _mongo_collection
    if _mongo_collection is not None:
        return _mongo_collection
    with _singleton_lock:
        if _mongo_collection is None:
            from pymongo import MongoClient

            uri = cfg.mongodb.get("uri", "mongodb://localhost:27017/")
            db_name = cfg.mongodb.get("database", "Rag")
            coll_name = cfg.mongodb.get("collection", "Document Storage")
            logger.info(
                "Initializing MongoDB connection db=%s collection=%s",
                db_name, coll_name,
            )
            client = MongoClient(uri)
            _mongo_collection = client[db_name][coll_name]
    return _mongo_collection
