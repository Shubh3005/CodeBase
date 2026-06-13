"""
Explanation Agent — builds a prompt from retrieved AST chunks and calls
llama-3.3-70b-versatile via Groq to produce a cited, human-readable answer.
"""
import logging
import re

from groq import Groq

from config import get_settings
from models.query import Citation

logger = logging.getLogger(__name__)
settings = get_settings()


def _groq_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


_SYSTEM_PROMPT = """You are CodeBase, an expert software engineer helping a new team member understand an unfamiliar codebase.

You are given a set of code chunks retrieved from the repository. Each chunk has a file path and line numbers.

Rules:
1. Answer the question clearly and concisely, assuming the reader is an experienced engineer new to THIS codebase.
2. Cite every factual claim with [file_path:line_start-line_end]. Use the exact paths provided in the context.
3. If the retrieved chunks are insufficient to answer fully, say so — never hallucinate code details.
4. Prefer architectural explanation over line-by-line commentary.
5. Format citations as [path/to/file.py:10-45] inline, not as a separate list.
"""


def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"--- Chunk {i}: {chunk.get('symbol_type', 'code')} `{chunk.get('symbol_name', '')}`"
            f" in {chunk['file_path']} (lines {chunk['start_line']}-{chunk['end_line']}) ---"
        )
        parts.append(f"{header}\n{chunk['raw_code']}")
    return "\n\n".join(parts)


def _parse_citations(answer: str, chunks: list[dict]) -> list[Citation]:
    """Extract [file:line-line] citations from the LLM response."""
    seen = set()
    citations = []

    pattern = re.compile(r"\[([^\]]+):(\d+)[–\-](\d+)\]")
    for match in pattern.finditer(answer):
        file_path = match.group(1)
        line_start = int(match.group(2))
        line_end = int(match.group(3))
        key = (file_path, line_start)
        if key in seen:
            continue
        seen.add(key)

        symbol_name = next(
            (c["symbol_name"] for c in chunks if c["file_path"] == file_path),
            "",
        )
        github_url = next(
            (c.get("github_url") for c in chunks if c["file_path"] == file_path),
            None,
        )
        citations.append(Citation(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            symbol_name=symbol_name,
            github_url=github_url,
        ))

    return citations


def answer(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> tuple[str, list[Citation], int]:
    """
    Generate an answer for the question given retrieved chunks.

    Returns (answer_text, citations, tokens_used).
    history is a list of {"role": "user"|"assistant", "content": str} dicts
    representing the last N turns of the conversation.
    """
    context_block = _build_context_block(chunks)
    user_message = f"CODEBASE CONTEXT:\n{context_block}\n\nQUESTION: {question}"

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_message})

    client = _groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=2048,
    )

    answer_text = response.choices[0].message.content
    tokens_used = response.usage.completion_tokens if response.usage else 0
    citations = _parse_citations(answer_text, chunks)

    return answer_text, citations, tokens_used
