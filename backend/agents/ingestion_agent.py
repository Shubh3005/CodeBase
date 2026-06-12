"""
Ingestion Agent — clones a repo, parses AST chunks, embeds with TF-IDF
(scikit-learn TfidfVectorizer), stores chunks in DynamoDB, and persists
the FAISS index + fitted vectorizer to S3.
"""
import logging
import os
import pickle
import shutil
import tempfile

import boto3
import faiss
import numpy as np
from git import Repo as GitRepo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import get_settings
from db import dynamo
from utils.ast_parser import iter_repo_chunks

logger = logging.getLogger(__name__)
settings = get_settings()

_BATCH_DYNAMO = 25  # DynamoDB BatchWriteItem limit


def _s3_client():
    return boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )


def _chunk_text(chunk: dict) -> str:
    parts = [
        f"File: {chunk['file_path']}",
        f"Symbol: {chunk['symbol_type']} {chunk['symbol_name']}",
    ]
    if chunk.get("docstring"):
        parts.append(f"Docstring: {chunk['docstring']}")
    parts.append(chunk["raw_code"])
    return "\n".join(parts)


def _save_to_s3(index: faiss.Index, vectorizer: TfidfVectorizer, repo_id: str) -> None:
    s3 = _s3_client()

    idx_tmp = tempfile.mktemp(suffix=".index")
    faiss.write_index(index, idx_tmp)
    s3.upload_file(idx_tmp, settings.s3_bucket, f"faiss/{repo_id}.index")
    os.unlink(idx_tmp)

    vec_tmp = tempfile.mktemp(suffix=".vectorizer")
    with open(vec_tmp, "wb") as f:
        pickle.dump(vectorizer, f)
    s3.upload_file(vec_tmp, settings.s3_bucket, f"faiss/{repo_id}.vectorizer")
    os.unlink(vec_tmp)


def run(
    repo_id: str,
    github_url: str,
    progress_callback=None,
) -> int:
    """Full ingestion pipeline. Returns total chunk count."""

    def report(step: str, pct: int):
        if progress_callback:
            progress_callback(step, pct)

    tmpdir = tempfile.mkdtemp()
    try:
        # ── 1. Clone ─────────────────────────────────────────────────────────
        report("cloning", 5)
        print(f"[ingestion:{repo_id}] Cloning {github_url} ...")
        GitRepo.clone_from(github_url, tmpdir, depth=1)
        print(f"[ingestion:{repo_id}] Clone complete.")
        report("cloning", 15)

        # ── 2. Parse AST ─────────────────────────────────────────────────────
        report("parsing", 20)
        print(f"[ingestion:{repo_id}] Parsing AST chunks ...")
        all_chunks = list(iter_repo_chunks(tmpdir, repo_id))
        print(f"[ingestion:{repo_id}] Parsed {len(all_chunks)} chunks.")

        # Compute GitHub deep-link for each chunk so the frontend can link directly
        github_base_url = github_url.rstrip("/")
        if github_base_url.endswith(".git"):
            github_base_url = github_base_url[:-4]
        for chunk in all_chunks:
            chunk["github_url"] = (
                f"{github_base_url}/blob/main/{chunk['file_path']}#L{chunk['start_line']}"
            )
        if not all_chunks:
            report("indexing", 100)
            return 0

        # ── 3. Embed (TF-IDF) ────────────────────────────────────────────────
        report("embedding", 30)
        total = len(all_chunks)
        texts = [_chunk_text(c) for c in all_chunks]

        print(f"[ingestion:{repo_id}] Fitting TF-IDF on {total} chunks (max_features={settings.embed_dimensions}) ...")
        vectorizer = TfidfVectorizer(max_features=settings.embed_dimensions, sublinear_tf=True)
        sparse_matrix = vectorizer.fit_transform(texts)  # (n_chunks, vocab_size)

        # Normalize rows for cosine similarity via inner product
        dense = normalize(sparse_matrix, norm="l2").toarray().astype(np.float32)
        actual_dim = dense.shape[1]
        print(f"[ingestion:{repo_id}] Embedding complete. Dimensions: {actual_dim}.")
        report("embedding", 65)

        for j, chunk in enumerate(all_chunks):
            chunk["embedding_id"] = str(j)

        index = faiss.IndexFlatIP(actual_dim)
        index.add(dense)
        print(f"[ingestion:{repo_id}] FAISS index built — {index.ntotal} vectors, dim={actual_dim}.")

        # ── 4. Store in DynamoDB ──────────────────────────────────────────────
        report("indexing", 72)
        print(f"[ingestion:{repo_id}] Writing {total} chunks to DynamoDB ...")
        for i in range(0, total, _BATCH_DYNAMO):
            dynamo.batch_put_chunks(all_chunks[i : i + _BATCH_DYNAMO])
        print(f"[ingestion:{repo_id}] DynamoDB write complete.")

        # ── 5. Persist FAISS index + vectorizer to S3 ─────────────────────────
        report("indexing", 90)
        print(f"[ingestion:{repo_id}] Saving FAISS index and vectorizer to S3 ...")
        _save_to_s3(index, vectorizer, repo_id)
        print(f"[ingestion:{repo_id}] S3 save complete.")

        report("indexing", 100)
        print(f"[ingestion:{repo_id}] Done — {total} chunks indexed.")
        return total

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
