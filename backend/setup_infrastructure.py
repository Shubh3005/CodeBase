"""
Run once to create AWS infrastructure and bootstrap the Aurora schema.

Usage:
    cd backend
    cp .env.example .env  # fill in real values first
    python setup_infrastructure.py
"""
import json
import os
import sys

import boto3
from botocore.exceptions import ClientError

# Load .env before importing config
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional here; set env vars manually

from config import get_settings
# import db.aurora as aurora_db  # Aurora disabled for DynamoDB/S3-only testing

settings = get_settings()


def create_dynamodb_table():
    client = boto3.client(
        "dynamodb",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )

    table_name = settings.dynamodb_table_ast_chunks
    try:
        client.describe_table(TableName=table_name)
        print(f"  [OK] DynamoDB table '{table_name}' already exists")
        return
    except client.exceptions.ResourceNotFoundException:
        pass

    print(f"  Creating DynamoDB table '{table_name}'...")
    client.create_table(
        TableName=table_name,
        AttributeDefinitions=[
            {"AttributeName": "repo_id",  "AttributeType": "S"},
            {"AttributeName": "chunk_id", "AttributeType": "S"},
        ],
        KeySchema=[
            {"AttributeName": "repo_id",  "KeyType": "HASH"},
            {"AttributeName": "chunk_id", "KeyType": "RANGE"},
        ],
        BillingMode="PAY_PER_REQUEST",  # on-demand — no capacity planning needed
    )

    waiter = client.get_waiter("table_exists")
    waiter.wait(TableName=table_name)
    print(f"  [OK] DynamoDB table '{table_name}' created")


def create_s3_bucket():
    client = boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )
    bucket = settings.s3_bucket
    try:
        client.head_bucket(Bucket=bucket)
        print(f"  [OK] S3 bucket '{bucket}' already exists")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise

    print(f"  Creating S3 bucket '{bucket}'...")
    if settings.aws_region == "us-east-1":
        client.create_bucket(Bucket=bucket)
    else:
        client.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": settings.aws_region},
        )

    # Block all public access
    client.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    print(f"  [OK] S3 bucket '{bucket}' created (public access blocked)")


# async def run_aurora_schema():  # Aurora disabled
#     print("  Connecting to Aurora and running schema.sql...")
#     await aurora_db.init_pool()
#     schema_path = os.path.join(os.path.dirname(__file__), "db", "schema.sql")
#     await aurora_db.run_schema(schema_path)
#     await aurora_db.close_pool()
#     print("  [OK] Aurora schema applied")


def main():
    print("\n=== CodeBase Infrastructure Setup ===\n")

    print("[1/2] DynamoDB")
    create_dynamodb_table()

    print("\n[2/2] S3")
    create_s3_bucket()

    # Aurora disabled — re-enable [3/3] when RDS is configured
    # print("\n[3/3] Aurora PostgreSQL schema")
    # if not settings.aurora_dsn:
    #     print("  [SKIP] AURORA_DSN not set — run manually after configuring RDS")
    # else:
    #     asyncio.run(run_aurora_schema())

    print("\n=== Setup complete ===\n")


if __name__ == "__main__":
    main()
