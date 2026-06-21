import os
# Must be set before any native library (faiss, torch, OpenMP) is imported.
# On macOS, FAISS and PyTorch each bundle an OpenMP runtime; without these
# the pthread mutex in abseil/glog deadlocks on first import.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import get_settings
# from db.aurora import init_pool, close_pool  # Aurora disabled for DynamoDB/S3-only testing
from routers import repos, queries
# from routers import auth, jobs  # Disabled: require Aurora PostgreSQL

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] sentence-transformers embeddings (lazy-loaded on first request).")
    # await init_pool()   # Aurora disabled
    yield
    # await close_pool()  # Aurora disabled


app = FastAPI(
    title="CodeBase API",
    description="AI-powered codebase onboarding — instant summaries and Q&A over any GitHub repo.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(auth.router, prefix="/api/auth", tags=["auth"])  # requires Aurora
# app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])   # requires Aurora
app.include_router(repos.router, prefix="/api/repos", tags=["repos"])
app.include_router(queries.router, prefix="/api/repos", tags=["queries"])


@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.environment}
