"""
Ingestion Agent — clones a repo, parses AST chunks, embeds with
sentence-transformers (all-MiniLM-L6-v2), stores chunks in DynamoDB,
and persists the FAISS index to S3.
"""
import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone

import boto3
import faiss
import numpy as np
from git import Repo as GitRepo
from sklearn.preprocessing import normalize

from config import get_settings
from db import dynamo
from utils.ast_parser import iter_repo_chunks

logger = logging.getLogger(__name__)
settings = get_settings()

_BATCH_DYNAMO = 25  # DynamoDB BatchWriteItem limit

_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".java": "Java",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".cpp": "C/C++", ".cc": "C/C++", ".c": "C/C++",
}

_SUMMARY_ENTRY_POINTS = frozenset({
    "main.py", "app.py", "index.py", "index.js", "index.ts",
    "app.js", "app.ts", "server.py", "run.py",
})

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # fixed by the model, not configurable via settings.embed_dimensions

# Module-level cache so the model is loaded once per process, not once per ingestion call.
_model_cache = {}


def _get_embedding_model():
    if "model" not in _model_cache:
        from sentence_transformers import SentenceTransformer  # lazy — avoids OMP deadlock at import time
        print(f"[ingestion] Loading embedding model {EMBEDDING_MODEL_NAME} ...")
        _model_cache["model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[ingestion] Embedding model loaded.")
    return _model_cache["model"]


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


def _save_to_s3(index: faiss.Index, repo_id: str, position_map: list[str]) -> None:
    """
    Persist the FAISS index and position map to S3.
    position_map[i] is the chunk_id for FAISS vector i, so retrieval never
    needs to scan DynamoDB just to resolve embedding positions.
    """
    s3 = _s3_client()

    fd, idx_tmp = tempfile.mkstemp(suffix=".index")
    os.close(fd)
    try:
        faiss.write_index(index, idx_tmp)
        s3.upload_file(idx_tmp, settings.s3_bucket, f"faiss/{repo_id}.index")
    finally:
        os.unlink(idx_tmp)

    s3.put_object(
        Bucket=settings.s3_bucket,
        Key=f"faiss/{repo_id}.position_map.json",
        Body=json.dumps(position_map).encode(),
        ContentType="application/json",
    )


def _generate_summary(repo_id: str, github_base_url: str, all_chunks: list[dict]) -> None:
    """Build a repo health summary from already-parsed chunks and upload it to S3."""
    repo_name = "/".join(github_base_url.rstrip("/").split("/")[-2:])

    file_paths = {c["file_path"] for c in all_chunks}

    lang_files: dict[str, set] = {}
    for fp in file_paths:
        ext = os.path.splitext(fp)[1].lower()
        lang = _EXT_TO_LANGUAGE.get(ext, "Other")
        lang_files.setdefault(lang, set()).add(fp)
    languages = {lang: len(files) for lang, files in lang_files.items()}

    entry_points = sorted(
        {os.path.basename(fp) for fp in file_paths if os.path.basename(fp) in _SUMMARY_ENTRY_POINTS}
    )

    file_chunk_counts = Counter(c["file_path"] for c in all_chunks)
    largest_files = [
        {"file_path": fp, "chunk_count": cnt}
        for fp, cnt in file_chunk_counts.most_common(3)
    ]

    summary = {
        "repo_name": repo_name,
        "total_files": len(file_paths),
        "total_chunks": len(all_chunks),
        "languages": languages,
        "entry_points": entry_points,
        "largest_files": largest_files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    _s3_client().put_object(
        Bucket=settings.s3_bucket,
        Key=f"faiss/{repo_id}.summary.json",
        Body=json.dumps(summary).encode(),
        ContentType="application/json",
    )
    print(f"[ingestion:{repo_id}] Health summary saved to S3.")


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
        git_repo = GitRepo.clone_from(github_url, tmpdir, depth=1)
        try:
            default_branch = git_repo.active_branch.name
        except TypeError:
            default_branch = "main"
        print(f"[ingestion:{repo_id}] Clone complete. Default branch: {default_branch}")
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
                f"{github_base_url}/blob/{default_branch}/{chunk['file_path']}#L{chunk['start_line']}"
            )
        if not all_chunks:
            report("indexing", 100)
            return 0

        # ── 3. Embed (sentence-transformers) ───────────────────────────────────
        report("embedding", 30)
        total = len(all_chunks)
        texts = [_chunk_text(c) for c in all_chunks]

        model = _get_embedding_model()
        print(f"[ingestion:{repo_id}] Embedding {total} chunks with {EMBEDDING_MODEL_NAME} "
              f"(dim={EMBEDDING_DIM}) ...")

        # batch_size controls memory/throughput; show_progress_bar off to keep logs clean
        raw_embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Normalize rows for cosine similarity via inner product (same as before)
        dense = normalize(raw_embeddings, norm="l2").astype(np.float32)
        actual_dim = dense.shape[1]
        print(f"[ingestion:{repo_id}] Embedding complete. Dimensions: {actual_dim}.")
        report("embedding", 65)

        for j, chunk in enumerate(all_chunks):
            chunk["embedding_id"] = str(j)

        # position_map[i] == chunk_id for FAISS vector i — built after embedding_ids are assigned
        position_map = [chunk["chunk_id"] for chunk in all_chunks]

        index = faiss.IndexFlatIP(actual_dim)
        index.add(dense)
        print(f"[ingestion:{repo_id}] FAISS index built — {index.ntotal} vectors, dim={actual_dim}.")

        # ── 4. Store in DynamoDB ──────────────────────────────────────────────
        report("indexing", 72)
        print(f"[ingestion:{repo_id}] Writing {total} chunks to DynamoDB ...")
        for i in range(0, total, _BATCH_DYNAMO):
            dynamo.batch_put_chunks(all_chunks[i : i + _BATCH_DYNAMO])
        print(f"[ingestion:{repo_id}] DynamoDB write complete.")

        # ── 5. Persist FAISS index + position map to S3 ──────────────────────
        report("indexing", 90)
        print(f"[ingestion:{repo_id}] Saving FAISS index and position map to S3 ...")
        _save_to_s3(index, repo_id, position_map)
        print(f"[ingestion:{repo_id}] S3 save complete.")

        # ── 6. Generate health summary ────────────────────────────────────────
        print(f"[ingestion:{repo_id}] Generating health summary ...")
        _generate_summary(repo_id, github_base_url, all_chunks)

        report("indexing", 100)
        print(f"[ingestion:{repo_id}] Done — {total} chunks indexed.")
        return total

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)