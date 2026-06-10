from fastapi import APIRouter, HTTPException
from models.repo import JobStatusResponse
from db import aurora

router = APIRouter()


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    row = await aurora.fetchrow(
        """
        SELECT j.id, j.repo_id, j.status, j.progress, j.step,
               j.chunk_count, j.error_message, j.started_at, j.completed_at
        FROM   ingestion_jobs j
        WHERE  j.id = $1
        """,
        job_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=str(row["id"]),
        repo_id=str(row["repo_id"]),
        status=row["status"],
        progress=row["progress"],
        step=row["step"],
        chunk_count=row["chunk_count"],
        error_message=row["error_message"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )
