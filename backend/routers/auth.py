from fastapi import APIRouter, HTTPException, status, Depends
from models.user import SignupRequest, LoginRequest, TokenResponse, UserResponse
from utils.auth_utils import hash_password, verify_password, create_access_token, get_current_user
from db import aurora

router = APIRouter()


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(body: SignupRequest):
    existing = await aurora.fetchrow("SELECT id FROM users WHERE email = $1", body.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    team_name = body.team_name or f"{body.email.split('@')[0]}'s team"

    user_row = await aurora.fetchrow(
        "INSERT INTO users (email, password_hash) VALUES ($1, $2) RETURNING id, email, created_at",
        body.email,
        hash_password(body.password),
    )
    team_row = await aurora.fetchrow(
        "INSERT INTO teams (name) VALUES ($1) RETURNING id",
        team_name,
    )
    await aurora.execute(
        "INSERT INTO team_members (user_id, team_id, role) VALUES ($1, $2, 'owner')",
        user_row["id"],
        team_row["id"],
    )
    await aurora.execute(
        "INSERT INTO usage_events (user_id, team_id, event_type) VALUES ($1, $2, 'signup')",
        user_row["id"],
        team_row["id"],
    )

    token = create_access_token(str(user_row["id"]), user_row["email"])
    return TokenResponse(access_token=token, user_id=str(user_row["id"]), email=user_row["email"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    row = await aurora.fetchrow(
        "SELECT id, email, password_hash FROM users WHERE email = $1",
        body.email,
    )
    if not row or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(str(row["id"]), row["email"])
    return TokenResponse(access_token=token, user_id=str(row["id"]), email=row["email"])


@router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    row = await aurora.fetchrow(
        "SELECT id, email, created_at FROM users WHERE id = $1",
        current_user["sub"],
    )
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(id=str(row["id"]), email=row["email"], created_at=row["created_at"])
