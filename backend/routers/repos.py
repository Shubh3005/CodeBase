import asyncio
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, BackgroundTasks, HTTPException

from agents import ingestion_agent, retrieval_agent
# from db import aurora  # Aurora disabled for DynamoDB/S3-only testing
from db import dynamo
from models.repo import IngestRequest, IngestResponse, JobStatusResponse
from config import get_settings
# from utils.auth_utils import get_current_user  # Auth disabled (no Aurora)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


def _s3():
    return boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )

_executor = ThreadPoolExecutor(max_workers=2)

_GITHUB_RE = re.compile(r"github\.com/([^/]+/[^/]+?)(?:\.git)?$")

# In-memory job store (replaces Aurora ingestion_jobs table)
_jobs: dict[str, dict] = {}


def _parse_repo_name(url: str) -> str:
    m = _GITHUB_RE.search(url)
    return m.group(1) if m else url.rstrip("/").split("/")[-1]


def _update_job(job_id: str, status: str, progress: int, step: str | None, chunk_count: int = 0, error: str | None = None):
    if job_id in _jobs:
        _jobs[job_id].update(
            status=status, progress=progress, step=step,
            chunk_count=chunk_count, error_message=error,
        )
        if status in ("COMPLETE", "FAILED"):
            _jobs[job_id]["completed_at"] = datetime.now(timezone.utc)


def _run_ingestion(repo_id: str, job_id: str, github_url: str):
    """Blocking ingestion run — executed in a thread pool."""

    def progress_callback(step: str, pct: int):
        _update_job(job_id, "PROCESSING", pct, step)

    try:
        chunk_count = ingestion_agent.run(
            repo_id=repo_id,
            github_url=github_url,
            progress_callback=progress_callback,
        )
        retrieval_agent.invalidate_index_cache(repo_id)
        _update_job(job_id, "COMPLETE", 100, "indexing", chunk_count)
    except Exception as exc:
        logger.exception("Ingestion failed for repo %s", repo_id)
        raw = str(exc)
        if "128" in raw or "authentication" in raw.lower() or "not found" in raw.lower():
            friendly = "Repository not found or is private. Check the URL and make sure the repo is public."
        elif "timeout" in raw.lower():
            friendly = "Clone timed out — the repository may be too large or the network is slow."
        else:
            friendly = f"Ingestion failed: {raw}"
        _update_job(job_id, "FAILED", 0, None, 0, friendly)


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_repo(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    # current_user: dict = Depends(get_current_user),  # Auth disabled
):
    github_url = body.github_url.strip().rstrip("/")
    if not _GITHUB_RE.search(github_url):
        raise HTTPException(status_code=422, detail="Must be a valid github.com URL")

    repo_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    _jobs[job_id] = {
        "job_id": job_id,
        "repo_id": repo_id,
        "status": "QUEUED",
        "progress": 0,
        "step": None,
        "chunk_count": 0,
        "error_message": None,
        "started_at": datetime.now(timezone.utc),
        "completed_at": None,
    }

    background_tasks.add_task(
        lambda: _executor.submit(_run_ingestion, repo_id, job_id, github_url)
    )

    return IngestResponse(job_id=job_id, repo_id=repo_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


@router.get("/{repo_id}/summary")
async def get_repo_summary(repo_id: str):
    try:
        resp = _s3().get_object(
            Bucket=settings.s3_bucket,
            Key=f"faiss/{repo_id}.summary.json",
        )
        return json.loads(resp["Body"].read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise HTTPException(status_code=404, detail="Summary not found")
        raise
