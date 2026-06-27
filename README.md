# CodeBase

> AI-powered codebase intelligence — drop any GitHub URL, get instant answers cited to the exact file and line.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://codebase-frontend.vercel.app)
[![Built with FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)](https://fastapi.tiangolo.com)
[![AWS DynamoDB](https://img.shields.io/badge/database-DynamoDB-FF9900)](https://aws.amazon.com/dynamodb)
[![Deployed on Vercel](https://img.shields.io/badge/frontend-Vercel-black)](https://vercel.com)

---

## What it does

Every engineer knows the feeling: you join a new project, clone the repo, and stare at thousands of files with zero context. CodeBase fixes that.

- **Semantic Q&A** — ask any question about a codebase in natural language
- **File:line citations** — every answer links to the exact source line on GitHub
- **Repo health summary** — language, file count, and entry points auto-detected on ingest
- **Entry point boost** — intelligently surfaces `main.py`/`app.py` when asked
- **Multi-repo comparison** — compare two codebases side-by-side in one query

**Try it:** [codebase-frontend.vercel.app](https://codebase-frontend.vercel.app)

---

## Architecture

```
User → Next.js (Vercel) → FastAPI (AWS ECS)
                               ├── GitHub (clone)
                               ├── sentence-transformers + FAISS (search)
                               ├── AWS DynamoDB (chunk metadata)
                               ├── AWS S3 (FAISS indexes)
                               └── Groq LLaMA 3.3 70B (generation)
```

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, Vercel |
| Backend | FastAPI, Python 3.11 |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Vector search | FAISS (IndexFlatIP) |
| Chunk metadata | AWS DynamoDB |
| FAISS indexes | AWS S3 |
| Generation | Groq LLaMA 3.3 70B |
| Deployment | AWS ECS + ECR, Docker |

---

## How it works

1. **Ingest** — clone the GitHub repo, walk the source tree, parse every file into AST chunks (function, class, module level)
2. **Embed** — generate 384-dim embeddings with sentence-transformers, build a FAISS index
3. **Store** — chunk metadata to DynamoDB, FAISS index to S3
4. **Query** — embed the user's question, retrieve top-k chunks via FAISS, pass to Groq for cited answer generation
5. **Compare** — run retrieval on two repos in parallel via `asyncio.gather`, split citations by repo boundary

---

## Running locally

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env  # fill in AWS + Groq credentials
uvicorn main:app --reload --port 8001

# Frontend
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8001 npm run dev
```

**Required env vars:**
```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
DYNAMODB_TABLE=codebase-ast-chunks
S3_BUCKET=codebase-faiss-indexes
GROQ_API_KEY=
CORS_ORIGINS=http://localhost:3000
```

---

## Built for H0 Hackathon

This project was built for the [H0: Hack the Zero Stack](https://h01.devpost.com/) hackathon using AWS DynamoDB and Vercel. Read the build writeup on [Medium](https://medium.com/p/0f3407ce2ff9).
