"""
Orchestrator — routes requests to the correct sub-agent and manages
cross-cutting concerns (session history, usage logging).
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from agents import explanation_agent, ingestion_agent, retrieval_agent
from models.query import Citation, QueryResponse

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)


async def _run_in_thread(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn, *args)


async def handle_compare(
    repo_id1: str,
    repo_id2: str,
    question: str,
    history: list[dict],
) -> tuple[str, list[Citation], list[Citation], int]:
    """
    Retrieve from both repos in parallel, then generate a single comparative answer.
    """
    (chunks1, chunks2), (name1, name2) = await asyncio.gather(
        asyncio.gather(
            _run_in_thread(retrieval_agent.retrieve, repo_id1, question, 3),
            _run_in_thread(retrieval_agent.retrieve, repo_id2, question, 3),
        ),
        asyncio.gather(
            _run_in_thread(ingestion_agent.get_repo_name, repo_id1),
            _run_in_thread(ingestion_agent.get_repo_name, repo_id2),
        ),
    )

    answer_text, citations1, citations2, tokens = await _run_in_thread(
        explanation_agent.compare_answer,
        question, chunks1, name1, chunks2, name2, history,
    )
    return answer_text, citations1, citations2, tokens


async def handle_query(
    repo_id: str,
    question: str,
    history: list[dict],
) -> tuple[str, list[Citation], int]:
    """
    Retrieve relevant chunks then generate a cited answer.
    Runs sync FAISS + Groq calls in a thread pool to stay non-blocking.
    """
    chunks = await _run_in_thread(retrieval_agent.retrieve, repo_id, question)

    if not chunks:
        return (
            "I couldn't find relevant code in this repository for your question. "
            "Try rephrasing or asking about a specific file or function name.",
            [],
            0,
        )

    answer_text, citations, tokens = await _run_in_thread(
        explanation_agent.answer, question, chunks, history
    )
    return answer_text, citations, tokens
