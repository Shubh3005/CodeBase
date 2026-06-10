from pydantic import BaseModel
from typing import Optional


class Citation(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    symbol_name: str
    github_url: Optional[str] = None  # deep link into GitHub source


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # omit to start a new session


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    session_id: str
    tokens_used: Optional[int] = None
