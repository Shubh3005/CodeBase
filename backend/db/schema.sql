-- CodeBase Aurora PostgreSQL schema
-- Run once against your Aurora cluster to bootstrap the database.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Users ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ── Teams ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS teams (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(255) NOT NULL,
    plan       VARCHAR(50)  DEFAULT 'free',  -- free | team | enterprise
    created_at TIMESTAMPTZ  DEFAULT NOW()
);

-- ── Team membership ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS team_members (
    user_id   UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id   UUID REFERENCES teams(id) ON DELETE CASCADE,
    role      VARCHAR(50) DEFAULT 'member',  -- owner | admin | member
    joined_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, team_id)
);

-- ── Repos (metadata only — AST chunks live in DynamoDB) ─────────────────────

CREATE TABLE IF NOT EXISTS repos (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id        UUID REFERENCES teams(id) ON DELETE CASCADE,
    github_url     VARCHAR(500) NOT NULL,
    repo_name      VARCHAR(255) NOT NULL,
    default_branch VARCHAR(100) DEFAULT 'main',
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (team_id, github_url)
);

-- ── Ingestion jobs ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id            UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id       UUID    REFERENCES repos(id) ON DELETE CASCADE,
    status        VARCHAR(50)  DEFAULT 'PENDING',  -- PENDING | PROCESSING | COMPLETE | FAILED
    progress      INT          DEFAULT 0,           -- 0–100
    step          VARCHAR(255),                     -- cloning | parsing | embedding | indexing
    chunk_count   INT          DEFAULT 0,
    error_message TEXT,
    started_at    TIMESTAMPTZ  DEFAULT NOW(),
    completed_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_repo_id ON ingestion_jobs(repo_id);

-- ── Sessions (Q&A conversation context) ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    repo_id     UUID REFERENCES repos(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_repo ON sessions(user_id, repo_id);

-- ── Session messages (last N kept per session for RAG context window) ────────

CREATE TABLE IF NOT EXISTS session_messages (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role       VARCHAR(20) NOT NULL,  -- user | assistant
    content    TEXT        NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_messages_session ON session_messages(session_id, created_at);

-- ── Usage events (billing / analytics) ──────────────────────────────────────

CREATE TABLE IF NOT EXISTS usage_events (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID REFERENCES users(id) ON DELETE SET NULL,
    team_id    UUID REFERENCES teams(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,  -- query | ingest | signup
    metadata   JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_usage_events_team ON usage_events(team_id, created_at);
