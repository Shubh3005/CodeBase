from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    team_name: Optional[str] = None  # creates a personal team if omitted


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime
