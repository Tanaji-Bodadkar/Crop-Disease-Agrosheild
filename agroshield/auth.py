"""Local account registration and authentication for the academic prototype."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(os.environ.get("AGROSHIELD_AUTH_DB", ROOT / "data" / "users.sqlite3"))
HASH_NAME = "sha256"
HASH_ITERATIONS = 420_000
SALT_BYTES = 16
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _initialize() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _normalized_email(email: str) -> str:
    normalized = email.strip().casefold()
    if not EMAIL_RE.fullmatch(normalized):
        raise ValueError("Enter a valid email address.")
    return normalized


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(HASH_NAME, password.encode("utf-8"), salt, HASH_ITERATIONS)


def register_user(full_name: str, email: str, password: str) -> dict[str, str]:
    """Create an account with a random salt and a slow password hash."""
    name = " ".join(full_name.split())
    if len(name) < 2 or len(name) > 80:
        raise ValueError("Enter your full name (2–80 characters).")
    normalized_email = _normalized_email(email)
    if len(password) < 10:
        raise ValueError("Use a password with at least 10 characters.")

    salt = secrets.token_bytes(SALT_BYTES)
    encoded_salt = salt.hex()
    encoded_hash = _password_hash(password, salt).hex()
    _initialize()
    try:
        with _connect() as connection:
            connection.execute(
                "INSERT INTO users (email, full_name, password_salt, password_hash) VALUES (?, ?, ?, ?)",
                (normalized_email, name, encoded_salt, encoded_hash),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with this email already exists. Sign in instead.") from exc
    return {"email": normalized_email, "full_name": name}


def authenticate_user(email: str, password: str) -> dict[str, str] | None:
    """Return the account profile on success, or None for invalid credentials."""
    try:
        normalized_email = _normalized_email(email)
    except ValueError:
        return None

    _initialize()
    with _connect() as connection:
        row = connection.execute(
            "SELECT email, full_name, password_salt, password_hash FROM users WHERE email = ?",
            (normalized_email,),
        ).fetchone()
    if row is None:
        return None

    try:
        salt = bytes.fromhex(row["password_salt"])
        expected = bytes.fromhex(row["password_hash"])
    except ValueError:
        return None
    if not hmac.compare_digest(_password_hash(password, salt), expected):
        return None
    return {"email": row["email"], "full_name": row["full_name"]}
