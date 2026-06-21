"""
Retrieval Agent — loads the FAISS index for a repo from S3, embeds the
query with sentence-transformers (all-MiniLM-L6-v2), runs nearest-neighbor
search, then batch-fetches the matching AST chunks from DynamoDB.
"""
import logging
import os
import tempfile

import boto3
import faiss
import numpy as np
from sklearn.preprocessing import normalize

from config import get_settings
from db import dynamo

logger = logging.getLogger(__name__)
settings = get_settings()

_INDEX_CACHE: dict[str, faiss.Index] = {}

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level cache — same pattern as ingestion_agent.py, so the model is
# only loaded once per process regardless of how many repos/queries hit it.
_model_cache = {}


def _get_embedding_model():
    if "model" not in _model_cache:
        from sentence_transformers import SentenceTransformer  # lazy — avoids OMP deadlock at import time
        print(f"[retrieval] Loading embedding model {EMBEDDING_MODEL_NAME} ...")
        _model_cache["model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[retrieval] Embedding model loaded.")
    return _model_cache["model"]


def _s3_client():
    return boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


def _load_index(repo_id: str) -> faiss.Index:
    if repo_id in _INDEX_CACHE:
        print(f"[retrieval:{repo_id}] FAISS index loaded from cache ({_INDEX_CACHE[repo_id].ntotal} vectors, dim={_INDEX_CACHE[repo_id].d})")
        return _INDEX_CACHE[repo_id]
    print(f"[retrieval:{repo_id}] Downloading FAISS index from S3 (bucket={settings.s3_bucket}) ...")
    tmp = tempfile.mktemp(suffix=".index")
    try:
        _s3_client().download_file(settings.s3_bucket, f"faiss/{repo_id}.index", tmp)
        index = faiss.read_index(tmp)
        _INDEX_CACHE[repo_id] = index
        print(f"[retrieval:{repo_id}] FAISS index loaded — {index.ntotal} vectors, dim={index.d}")
        return index
    except Exception as e:
        print(f"[retrieval:{repo_id}] ERROR loading FAISS index: {e}")
        raise
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def invalidate_index_cache(repo_id: str) -> None:
    _INDEX_CACHE.pop(repo_id, None)


def embed_query(question: str) -> np.ndarray:
    model = _get_embedding_model()
    raw = model.encode([question], convert_to_numpy=True)
    dense = normalize(raw, norm="l2").astype(np.float32)
    print(f"[retrieval] Query vector shape: {dense.shape}, norm: {float(np.linalg.norm(dense)):.4f}")
    return dense


def retrieve(repo_id: str, question: str, top_k: int = 8) -> list[dict]:
    """
    Embed the question with sentence-transformers, search FAISS, and
    return the top_k AST chunks with full metadata from DynamoDB.
    """
    print(f"[retrieval:{repo_id}] Query: {question!r}")
    index = _load_index(repo_id)
    query_vec = embed_query(question)

    k = min(top_k, index.ntotal)
    if k == 0:
        print(f"[retrieval:{repo_id}] FAISS index is empty — nothing to search")
        return []

    scores, indices = index.search(query_vec, k)
    print(f"[retrieval:{repo_id}] FAISS top-{k} scores: {scores[0].tolist()}")
    print(f"[retrieval:{repo_id}] FAISS top-{k} indices: {indices[0].tolist()}")

    all_chunks = dynamo.list_chunks_for_repo(repo_id)
    print(f"[retrieval:{repo_id}] DynamoDB returned {len(all_chunks)} chunks total; "
          f"{sum(1 for c in all_chunks if 'embedding_id' in c)} have embedding_id")
    position_map = {c["embedding_id"]: c["chunk_id"] for c in all_chunks if "embedding_id" in c}

    hit_ids = [
        position_map[str(pos)]
        for pos in indices[0]
        if str(pos) in position_map
    ]
    print(f"[retrieval:{repo_id}] Mapped {len(hit_ids)}/{k} FAISS hits to chunk_ids")

    if not hit_ids:
        print(f"[retrieval:{repo_id}] No chunk_ids resolved — check embedding_id alignment between FAISS and DynamoDB")
        return []

    full_chunks = dynamo.batch_get_chunks(repo_id, hit_ids)
    print(f"[retrieval:{repo_id}] batch_get_chunks returned {len(full_chunks)} chunks")
    return full_chunks