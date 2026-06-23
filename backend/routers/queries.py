import uuid

from fastapi import APIRouter, HTTPException

from agents.orchestrator import handle_compare, handle_query
from models.query import CompareRequest, CompareResponse, QueryRequest, QueryResponse

router = APIRouter()

_MAX_HISTORY_MESSAGES = 12  # 6 turns × 2 (user + assistant)

# In-memory session store: session_id → list of {"role": ..., "content": ...}
_sessions: dict[str, list[dict]] = {}


@router.post("/compare", response_model=CompareResponse)
async def compare_repos(body: CompareRequest):
    session_id = body.session_id or str(uuid.uuid4())

    if body.session_id and body.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    history = _sessions.get(session_id, [])

    answer_text, citations1, citations2, tokens_used = await handle_compare(
        repo_id1=body.repo_ids[0],
        repo_id2=body.repo_ids[1],
        question=body.question,
        history=history,
    )

    updated = history + [
        {"role": "user", "content": body.question},
        {"role": "assistant", "content": answer_text},
    ]
    _sessions[session_id] = updated[-_MAX_HISTORY_MESSAGES:]

    return CompareResponse(
        answer=answer_text,
        citations_repo1=citations1,
        citations_repo2=citations2,
        session_id=session_id,
        tokens_used=tokens_used,
    )


@router.post("/{repo_id}/query", response_model=QueryResponse)
async def query_repo(repo_id: str, body: QueryRequest):
    session_id = body.session_id or str(uuid.uuid4())

    if body.session_id and body.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    history = _sessions.get(session_id, [])

    answer_text, citations, tokens_used = await handle_query(
        repo_id=repo_id,
        question=body.question,
        history=history,
    )

    # Persist this turn and trim to the sliding window
    updated = history + [
        {"role": "user", "content": body.question},
        {"role": "assistant", "content": answer_text},
    ]
    _sessions[session_id] = updated[-_MAX_HISTORY_MESSAGES:]

    return QueryResponse(
        answer=answer_text,
        citations=citations,
        session_id=session_id,
        tokens_used=tokens_used,
    )
