from pydantic import BaseModel, HttpUrl
from datetime import datetime
from typing import Optional
import uuid


class IngestRequest(BaseModel):
    github_url: str
    team_id: Optional[str] = None


class IngestResponse(BaseModel):
    job_id: str
    repo_id: str
    status: str = "PENDING"


class JobStatusResponse(BaseModel):
    job_id: str
    repo_id: str
    status: str          # PENDING | PROCESSING | COMPLETE | FAILED
    progress: int        # 0–100
    step: Optional[str]  # cloning | parsing | embedding | indexing
    chunk_count: int
    error_message: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]


class ModuleSummary(BaseModel):
    file_path: str
    summary: str
    symbol_count: int


class RepoSummaryResponse(BaseModel):
    repo_id: str
    repo_name: str
    modules: list[ModuleSummary]


class ASTChunk(BaseModel):
    repo_id: str
    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_type: str    # function | class | method | async_function
    raw_code: str
    docstring: Optional[str]
    start_line: int
    end_line: int
    embedding_id: Optional[str] = None  # maps to FAISS internal index position
