from pydantic import BaseModel, Field
from typing import Optional


class Citation(BaseModel):
    file_path: str
    line_start: int
    line_end: int
    symbol_name: str
    github_url: Optional[str] = None  # deep link into GitHub source


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: Optional[str] = None  # omit to start a new session


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    session_id: str
    tokens_used: Optional[int] = None
