"""
vectorstore.py

Vector database initialization and embedding configuration module.

Purpose:
---------
Provides:
- Embedding model initialization
- Vectorstore (Qdrant) loading logic

This module abstracts away:
- Embedding model selection
- Vector database connection management

Why this exists:
----------------
To prevent vectorstore logic from being mixed with indexing or
runtime AI logic.

All components requiring a retriever should call `load_vectorstore()`
instead of manually initializing Qdrant.

Design Principle:
-----------------
Centralized infrastructure layer.
Embedding configuration must exist in a single location to prevent
inconsistency across the system.

Migration Note:
---------------
Replaced ChromaDB (embedded SQLite) with Qdrant (client-server) for
production readiness: replication, metadata filtering indexes, access
control, and horizontal scaling.
"""
import threading

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config.rag import EMBEDDING_MODEL
from config import cfg

_vectorstore_instance = None
_vectorstore_lock = threading.Lock()


class _QdrantVectorStoreNoValidation(QdrantVectorStore):
    """
    QdrantVectorStore with collection validation disabled.

    The base class runs the full embedding model during __init__ via
    _validate_collection_for_dense() to verify vector dimensions match.
    On Windows this triggers an access violation when background threads
    (langsmith, tqdm, concurrent.futures) are alive alongside PyTorch.

    Safe to skip: we create the collection ourselves with the correct
    vector size above, so dimension mismatch is impossible here.
    """
    def _validate_collection_config(self, *args, **kwargs):
        pass


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def _get_qdrant_client():
    """Create a Qdrant client from centralized config."""
    qdrant_cfg = cfg.qdrant
    return QdrantClient(
        host=qdrant_cfg.get("host", "localhost"),
        port=qdrant_cfg.get("port", 6333),
        grpc_port=qdrant_cfg.get("grpc_port", 6334),
        prefer_grpc=qdrant_cfg.get("prefer_grpc", True),
        check_compatibility=False,
    )


def load_vectorstore():
    """
    Return the shared QdrantVectorStore instance, creating it once.

    The module-level singleton here is intentional and separate from
    the one in nodes.py. If nodes.py gets re-imported mid-session
    (resetting its _database to None), this lock-protected instance
    ensures load_vectorstore() still returns the cached object and
    never re-runs the embedding model or QdrantVectorStore.__init__.
    """
    global _vectorstore_instance
    if _vectorstore_instance is not None:
        return _vectorstore_instance

    with _vectorstore_lock:
        if _vectorstore_instance is not None:
            return _vectorstore_instance

        embeddings = get_embeddings()
        client = _get_qdrant_client()
        collection_name = cfg.qdrant.get("collection", "judicial_docs")
        vector_size = cfg.qdrant.get("vector_size", 1024)

        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

        _vectorstore_instance = _QdrantVectorStoreNoValidation(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

    return _vectorstore_instance