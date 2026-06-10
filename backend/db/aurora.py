import asyncpg
from config import get_settings

settings = get_settings()

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.aurora_dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized — call init_pool() at startup.")
    return _pool


async def fetchrow(query: str, *args):
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetch(query: str, *args) -> list:
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def execute(query: str, *args) -> str:
    pool = get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def run_schema(schema_path: str) -> None:
    """Bootstrap the database from schema.sql — safe to re-run (uses IF NOT EXISTS)."""
    with open(schema_path) as f:
        sql = f.read()
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(sql)
