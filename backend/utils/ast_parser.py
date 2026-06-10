import ast
import re
import uuid
from pathlib import Path
from typing import Iterator


def _extract_docstring(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value.strip()
    return None


def parse_python_file(file_path: str, source: str, repo_id: str) -> list[dict]:
    """Parse a Python source file and return one chunk per function/class."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    chunks = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start = node.lineno - 1  # 0-indexed
        end = node.end_lineno    # exclusive upper bound

        symbol_type = (
            "class" if isinstance(node, ast.ClassDef)
            else "async_function" if isinstance(node, ast.AsyncFunctionDef)
            else "function"
        )

        raw_code = "\n".join(lines[start:end])
        docstring = _extract_docstring(node)

        chunks.append({
            "repo_id": repo_id,
            "chunk_id": str(uuid.uuid4()),
            "file_path": file_path,
            "symbol_name": node.name,
            "symbol_type": symbol_type,
            "raw_code": raw_code,
            "docstring": docstring or "",
            "start_line": node.lineno,
            "end_line": node.end_lineno,
        })

    return chunks


_JS_TS_FUNCTION_RE = re.compile(
    r"(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
    re.MULTILINE,
)


def parse_js_ts_file(file_path: str, source: str, repo_id: str) -> list[dict]:
    """Regex-based chunking for JS/TS — splits on top-level function declarations."""
    lines = source.splitlines()
    chunks = []

    for match in _JS_TS_FUNCTION_RE.finditer(source):
        symbol_name = match.group(1) or match.group(2) or "anonymous"
        start_line = source[: match.start()].count("\n") + 1

        # Grab ~30 lines as the chunk body (no full AST for JS in stdlib)
        end_line = min(start_line + 30, len(lines))
        raw_code = "\n".join(lines[start_line - 1 : end_line])

        chunks.append({
            "repo_id": repo_id,
            "chunk_id": str(uuid.uuid4()),
            "file_path": file_path,
            "symbol_name": symbol_name,
            "symbol_type": "function",
            "raw_code": raw_code,
            "docstring": "",
            "start_line": start_line,
            "end_line": end_line,
        })

    return chunks


_SUPPORTED = {
    ".py": parse_python_file,
    ".ts": parse_js_ts_file,
    ".tsx": parse_js_ts_file,
    ".js": parse_js_ts_file,
    ".jsx": parse_js_ts_file,
}

_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", "coverage",
}


def iter_repo_chunks(repo_path: str, repo_id: str) -> Iterator[dict]:
    """Walk a cloned repo and yield AST chunks for all supported source files."""
    root = Path(repo_path)
    for path in root.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue

        parser = _SUPPORTED.get(path.suffix)
        if parser is None:
            continue

        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        relative = str(path.relative_to(root))
        yield from parser(relative, source, repo_id)
