"""
Authentication and Role-Based Access Control (RBAC) Module.
Implements:
- Bcrypt password hashing
- JWT encode/decode with 60-minute expiry
- In-memory rate limiting (max 5 failed attempts per IP per 5 minutes)
- FastAPI role-guard dependencies (Admin / Moderator)
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# JWT Secrets & Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-insecure-secret-key-replace-in-production")
if os.getenv("ENV", "development") == "production" and JWT_SECRET_KEY.startswith("dev-"):
    logger.critical("FATAL: Insecure default JWT_SECRET_KEY detected in production environment!")
    raise RuntimeError("JWT_SECRET_KEY must be set to a secure string in production.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 Bearer token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# In-memory IP rate limiter for login protection
# Key: client_ip -> List of failure timestamps
FAILED_LOGIN_ATTEMPTS = {}
RATE_LIMIT_WINDOW_SECONDS = 300  # 5 minutes
MAX_FAILED_ATTEMPTS = 5

def check_rate_limit(client_ip: str):
    """Verifies that an IP has not exceeded the max failed login threshold."""
    now = time.time()
    attempts = FAILED_LOGIN_ATTEMPTS.get(client_ip, [])
    # Filter out attempts older than window
    recent_attempts = [t for t in attempts if now - t < RATE_LIMIT_WINDOW_SECONDS]
    FAILED_LOGIN_ATTEMPTS[client_ip] = recent_attempts
    
    if len(recent_attempts) >= MAX_FAILED_ATTEMPTS:
        retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - recent_attempts[0]))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Account locked temporarily. Try again in {max(1, retry_after)} seconds."
        )

def record_failed_attempt(client_ip: str):
    now = time.time()
    attempts = FAILED_LOGIN_ATTEMPTS.get(client_ip, [])
    attempts.append(now)
    FAILED_LOGIN_ATTEMPTS[client_ip] = attempts

def reset_rate_limit(client_ip: str):
    if client_ip in FAILED_LOGIN_ATTEMPTS:
        del FAILED_LOGIN_ATTEMPTS[client_ip]

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials or token expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependency to get current user from token
def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme)
) -> dict:
    # Also check Authorization header directly in case custom fetch sent it
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token missing. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    email: str = payload.get("sub")
    role: str = payload.get("role")
    user_id: str = payload.get("user_id")

    if email is None or role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload structure.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "id": user_id,
        "email": email,
        "role": role
    }

def require_role(allowed_roles: List[str]):
    """Returns a dependency function verifying that the current user has an authorized role."""
    def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires one of roles: {allowed_roles}. Current role: {current_user['role']}"
            )
        return current_user
    return role_checker
