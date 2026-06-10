import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from config import get_settings

settings = get_settings()

_resource = None
_table = None


def get_table():
    global _resource, _table
    if _table is None:
        _resource = boto3.resource(
            "dynamodb",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        _table = _resource.Table(settings.dynamodb_table_ast_chunks)
    return _table


def put_chunk(chunk: dict) -> None:
    """Write a single AST chunk. chunk must have repo_id and chunk_id keys."""
    get_table().put_item(Item=chunk)


def batch_put_chunks(chunks: list[dict]) -> None:
    """Write up to 25 chunks in a single BatchWriteItem call."""
    table = get_table()
    with table.batch_writer() as batch:
        for chunk in chunks:
            batch.put_item(Item=chunk)


def get_chunk(repo_id: str, chunk_id: str) -> dict | None:
    resp = get_table().get_item(Key={"repo_id": repo_id, "chunk_id": chunk_id})
    return resp.get("Item")


def batch_get_chunks(repo_id: str, chunk_ids: list[str]) -> list[dict]:
    """Fetch up to 100 chunks by their chunk_ids for a given repo."""
    if not chunk_ids:
        return []

    resource = boto3.resource(
        "dynamodb",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )
    table_name = settings.dynamodb_table_ast_chunks
    keys = [{"repo_id": repo_id, "chunk_id": cid} for cid in chunk_ids]

    response = resource.batch_get_item(
        RequestItems={table_name: {"Keys": keys}}
    )
    return response["Responses"].get(table_name, [])


def list_chunks_for_repo(repo_id: str) -> list[dict]:
    """Return all chunk metadata for a repo (excluding raw_code for summary view)."""
    table = get_table()
    items = []
    kwargs = {
        "KeyConditionExpression": Key("repo_id").eq(repo_id),
        "ProjectionExpression": "chunk_id, file_path, symbol_name, symbol_type, docstring, start_line, end_line, embedding_id",
    }
    while True:
        resp = table.query(**kwargs)
        items.extend(resp["Items"])
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


def delete_chunks_for_repo(repo_id: str) -> int:
    """Delete all chunks for a repo. Returns deleted count."""
    table = get_table()
    chunks = list_chunks_for_repo(repo_id)
    with table.batch_writer() as batch:
        for chunk in chunks:
            batch.delete_item(Key={"repo_id": repo_id, "chunk_id": chunk["chunk_id"]})
    return len(chunks)
