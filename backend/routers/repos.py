import asyncio
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from agents import ingestion_agent
# from db import aurora  # Aurora disabled for DynamoDB/S3-only testing
from db import dynamo
from models.repo import IngestRequest, IngestResponse, JobStatusResponse, ModuleSummary, RepoSummaryResponse
# from utils.auth_utils import get_current_user  # Auth disabled (no Aurora)

logger = logging.getLogger(__name__)
router = APIRouter()

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


@router.get("/{repo_id}/summary", response_model=RepoSummaryResponse)
async def get_repo_summary(repo_id: str):
    chunks = dynamo.list_chunks_for_repo(repo_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Repo not found or not yet ingested")

    files: dict[str, list] = {}
    for c in chunks:
        files.setdefault(c["file_path"], []).append(c)

    modules = [
        ModuleSummary(
            file_path=fp,
            summary=f"{len(cs)} symbol(s): {', '.join(c['symbol_name'] for c in cs[:5])}{'…' if len(cs) > 5 else ''}",
            symbol_count=len(cs),
        )
        for fp, cs in sorted(files.items())
    ]

    return RepoSummaryResponse(
        repo_id=repo_id,
        repo_name=repo_id,
        modules=modules,
    )
