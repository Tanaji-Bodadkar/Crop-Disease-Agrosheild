"""Email-verified local accounts for the Agrosheild academic prototype."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import smtplib
import sqlite3
import ssl
import time
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(os.environ.get("AGROSHIELD_AUTH_DB", ROOT / "data" / "users.sqlite3"))
HASH_ITERATIONS = 420_000
SALT_BYTES = 16
CODE_TTL_SECONDS = 10 * 60
RESEND_WAIT_SECONDS = 60
MAX_CODE_ATTEMPTS = 5
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class EmailVerificationError(RuntimeError):
    """Raised when an email verification message cannot be delivered."""


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
                email_verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
        if "email_verified" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_verifications (
                email TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                password_salt TEXT,
                password_hash TEXT,
                purpose TEXT NOT NULL,
                code_salt TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                sent_at INTEGER NOT NULL
            )
            """
        )


def _normalized_email(email: str) -> str:
    normalized = email.strip().casefold()
    if len(normalized) > 254 or not EMAIL_RE.fullmatch(normalized):
        raise ValueError("Enter a valid email address.")
    return normalized


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, HASH_ITERATIONS)


def _email_config() -> dict[str, str | int]:
    try:
        import streamlit as st

        config = st.secrets.get("email", {})
    except Exception:
        config = {}
    values = {
        "smtp_host": config.get("smtp_host") or os.getenv("AGROSHIELD_SMTP_HOST"),
        "smtp_port": config.get("smtp_port") or os.getenv("AGROSHIELD_SMTP_PORT"),
        "smtp_username": config.get("smtp_username") or os.getenv("AGROSHIELD_SMTP_USERNAME"),
        "smtp_password": config.get("smtp_password") or os.getenv("AGROSHIELD_SMTP_PASSWORD"),
        "from_email": config.get("from_email") or os.getenv("AGROSHIELD_FROM_EMAIL"),
    }
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise EmailVerificationError(
            "Email verification is not configured yet. Add the SMTP settings described in the README's email setup section."
        )
    try:
        values["smtp_port"] = int(values["smtp_port"])
    except (TypeError, ValueError) as exc:
        raise EmailVerificationError("The SMTP port setting must be a number.") from exc
    return values


def _send_code(email: str, full_name: str, code: str) -> None:
    config = _email_config()
    message = EmailMessage()
    message["Subject"] = "Your Agrosheild verification code"
    message["From"] = str(config["from_email"])
    message["To"] = email
    message.set_content(
        f"Hello {full_name},\n\n"
        f"Your Agrosheild verification code is: {code}\n\n"
        "It expires in 10 minutes. If you did not request an account, you can ignore this email.\n\n"
        "Agrosheild"
    )
    try:
        host = str(config["smtp_host"])
        port = int(config["smtp_port"])
        context = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as server:
                server.login(str(config["smtp_username"]), str(config["smtp_password"]))
                server.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(str(config["smtp_username"]), str(config["smtp_password"]))
                server.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise EmailVerificationError(
            "We couldn't send the verification email. Check the SMTP settings and try again."
        ) from exc


def _code_hash(code: str, salt: bytes) -> bytes:
    return hashlib.sha256(salt + code.encode("ascii")).digest()


def _start_code(email: str, full_name: str, purpose: str, password_salt: str | None, password_hash: str | None) -> None:
    now = int(time.time())
    code = f"{secrets.randbelow(1_000_000):06d}"
    salt = secrets.token_bytes(SALT_BYTES)
    _initialize()
    with _connect() as connection:
        previous = connection.execute(
            "SELECT sent_at FROM pending_verifications WHERE email = ?", (email,)
        ).fetchone()
        if previous and now - previous["sent_at"] < RESEND_WAIT_SECONDS:
            wait = RESEND_WAIT_SECONDS - (now - previous["sent_at"])
            raise EmailVerificationError(f"Please wait {wait} seconds before requesting another code.")
        connection.execute(
            """
            INSERT INTO pending_verifications (
                email, full_name, password_salt, password_hash, purpose,
                code_salt, code_hash, expires_at, attempts, sent_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(email) DO UPDATE SET
                full_name=excluded.full_name,
                password_salt=excluded.password_salt,
                password_hash=excluded.password_hash,
                purpose=excluded.purpose,
                code_salt=excluded.code_salt,
                code_hash=excluded.code_hash,
                expires_at=excluded.expires_at,
                attempts=0,
                sent_at=excluded.sent_at
            """,
            (
                email,
                full_name,
                password_salt,
                password_hash,
                purpose,
                salt.hex(),
                _code_hash(code, salt).hex(),
                now + CODE_TTL_SECONDS,
                now,
            ),
        )
    try:
        _send_code(email, full_name, code)
    except EmailVerificationError:
        with _connect() as connection:
            connection.execute("DELETE FROM pending_verifications WHERE email = ?", (email,))
        raise


def begin_registration(full_name: str, email: str, password: str) -> str:
    name = " ".join(full_name.split())
    if len(name) < 2 or len(name) > 80:
        raise ValueError("Enter your full name (2–80 characters).")
    normalized_email = _normalized_email(email)
    if len(password) < 10:
        raise ValueError("Use a password with at least 10 characters.")
    _initialize()
    with _connect() as connection:
        existing = connection.execute(
            "SELECT full_name, email_verified FROM users WHERE email = ?", (normalized_email,)
        ).fetchone()
    if existing and existing["email_verified"]:
        raise ValueError("An account with this email already exists. Sign in instead.")
    salt = secrets.token_bytes(SALT_BYTES)
    _start_code(normalized_email, name, "register", salt.hex(), _password_hash(password, salt).hex())
    return normalized_email


def begin_existing_verification(email: str, password: str) -> dict[str, str] | None:
    """Send a code to a matching unverified account; return a verified profile if ready."""
    try:
        normalized_email = _normalized_email(email)
    except ValueError:
        return None
    _initialize()
    with _connect() as connection:
        row = connection.execute(
            "SELECT email, full_name, password_salt, password_hash, email_verified FROM users WHERE email = ?",
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
    profile = {"email": row["email"], "full_name": row["full_name"]}
    if row["email_verified"]:
        return profile
    _start_code(normalized_email, row["full_name"], "activate_existing", None, None)
    return None


def resend_verification(email: str) -> None:
    normalized_email = _normalized_email(email)
    _initialize()
    now = int(time.time())
    with _connect() as connection:
        row = connection.execute(
            "SELECT full_name, purpose, password_salt, password_hash, sent_at FROM pending_verifications WHERE email = ?",
            (normalized_email,),
        ).fetchone()
    if row is None:
        raise ValueError("Start account creation or sign-in again to request a new code.")
    if now - row["sent_at"] < RESEND_WAIT_SECONDS:
        raise ValueError(f"Please wait {RESEND_WAIT_SECONDS - (now - row['sent_at'])} seconds before requesting another code.")
    _start_code(normalized_email, row["full_name"], row["purpose"], row["password_salt"], row["password_hash"])


def verify_email_code(email: str, code: str) -> dict[str, str]:
    normalized_email = _normalized_email(email)
    normalized_code = code.strip()
    if not re.fullmatch(r"\d{6}", normalized_code):
        raise ValueError("Enter the six-digit code from the email.")
    _initialize()
    with _connect() as connection:
        pending = connection.execute(
            "SELECT * FROM pending_verifications WHERE email = ?", (normalized_email,)
        ).fetchone()
    if pending is None:
        raise ValueError("No active verification was found. Create the account or sign in again to request a new code.")
    now = int(time.time())
    if pending["expires_at"] < now:
        with _connect() as connection:
            connection.execute("DELETE FROM pending_verifications WHERE email = ?", (normalized_email,))
        raise ValueError("That code has expired. Request a new verification code.")
    if pending["attempts"] >= MAX_CODE_ATTEMPTS:
        with _connect() as connection:
            connection.execute("DELETE FROM pending_verifications WHERE email = ?", (normalized_email,))
        raise ValueError("Too many incorrect attempts. Start verification again to receive a new code.")
    salt = bytes.fromhex(pending["code_salt"])
    if not hmac.compare_digest(_code_hash(normalized_code, salt), bytes.fromhex(pending["code_hash"])):
        with _connect() as connection:
            connection.execute(
                "UPDATE pending_verifications SET attempts = attempts + 1 WHERE email = ?", (normalized_email,)
            )
        remaining = MAX_CODE_ATTEMPTS - pending["attempts"] - 1
        raise ValueError(f"That code doesn't match. {remaining} attempt(s) remaining.")

    if pending["purpose"] == "register":
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO users (email, full_name, password_salt, password_hash, email_verified)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(email) DO UPDATE SET
                    full_name=excluded.full_name,
                    password_salt=excluded.password_salt,
                    password_hash=excluded.password_hash,
                    email_verified=1
                """,
                (
                    normalized_email,
                    pending["full_name"],
                    pending["password_salt"],
                    pending["password_hash"],
                ),
            )
    else:
        with _connect() as connection:
            connection.execute("UPDATE users SET email_verified = 1 WHERE email = ?", (normalized_email,))
    with _connect() as connection:
        connection.execute("DELETE FROM pending_verifications WHERE email = ?", (normalized_email,))
    return {"email": normalized_email, "full_name": pending["full_name"]}


def authenticate_user(email: str, password: str) -> dict[str, str | bool] | None:
    try:
        normalized_email = _normalized_email(email)
    except ValueError:
        return None
    _initialize()
    with _connect() as connection:
        row = connection.execute(
            "SELECT email, full_name, password_salt, password_hash, email_verified FROM users WHERE email = ?",
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
    return {"email": row["email"], "full_name": row["full_name"], "email_verified": bool(row["email_verified"])}
