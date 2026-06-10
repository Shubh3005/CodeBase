"""
Retrieval Agent — loads the FAISS index and fitted TF-IDF vectorizer for a
repo from S3, embeds the query, runs nearest-neighbor search, then
batch-fetches the matching AST chunks from DynamoDB.
"""
import logging
import os
import pickle
import tempfile

import boto3
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import get_settings
from db import dynamo

logger = logging.getLogger(__name__)
settings = get_settings()

_INDEX_CACHE: dict[str, faiss.Index] = {}
_VECTORIZER_CACHE: dict[str, TfidfVectorizer] = {}


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


def _load_vectorizer(repo_id: str) -> TfidfVectorizer:
    if repo_id in _VECTORIZER_CACHE:
        v = _VECTORIZER_CACHE[repo_id]
        print(f"[retrieval:{repo_id}] Vectorizer loaded from cache (vocab size={len(v.vocabulary_)})")
        return v
    print(f"[retrieval:{repo_id}] Downloading vectorizer from S3 ...")
    tmp = tempfile.mktemp(suffix=".vectorizer")
    try:
        _s3_client().download_file(settings.s3_bucket, f"faiss/{repo_id}.vectorizer", tmp)
        with open(tmp, "rb") as f:
            vectorizer = pickle.load(f)
        _VECTORIZER_CACHE[repo_id] = vectorizer
        print(f"[retrieval:{repo_id}] Vectorizer loaded (vocab size={len(vectorizer.vocabulary_)})")
        return vectorizer
    except Exception as e:
        print(f"[retrieval:{repo_id}] ERROR loading vectorizer: {e}")
        raise
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def invalidate_index_cache(repo_id: str) -> None:
    _INDEX_CACHE.pop(repo_id, None)
    _VECTORIZER_CACHE.pop(repo_id, None)


def embed_query(question: str, vectorizer: TfidfVectorizer) -> np.ndarray:
    sparse = vectorizer.transform([question])
    dense = normalize(sparse, norm="l2").toarray().astype(np.float32)
    print(f"[retrieval] Query vector shape: {dense.shape}, non-zero dims: {int((dense != 0).sum())}, norm: {float(np.linalg.norm(dense)):.4f}")
    return dense


def retrieve(repo_id: str, question: str, top_k: int = 8) -> list[dict]:
    """
    Embed the question with the repo's fitted TF-IDF vectorizer, search
    FAISS, and return the top_k AST chunks with full metadata from DynamoDB.
    """
    print(f"[retrieval:{repo_id}] Query: {question!r}")
    index = _load_index(repo_id)
    vectorizer = _load_vectorizer(repo_id)
    query_vec = embed_query(question, vectorizer)

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
