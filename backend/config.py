from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # AWS
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # Aurora PostgreSQL
    aurora_dsn: str = ""

    # DynamoDB
    dynamodb_table_ast_chunks: str = "codebase-ast-chunks"

    # S3
    s3_bucket: str = "codebase-faiss-indexes"

    # Auth
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 10080  # 7 days

    # Groq
    groq_api_key: str = ""

    # Embeddings (TF-IDF max_features cap — actual FAISS dim derived at fit time)
    embed_dimensions: int = 8192

    # App
    environment: str = "development"
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
