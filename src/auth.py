"""
User Authentication
===================
Simple file-based auth for demo purposes.
Passwords hashed with bcrypt.
No database needed — JSON file storage.
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
import bcrypt

logger   = logging.getLogger(__name__)
AUTH_FILE = Path(__file__).parent.parent / "data" / "users.json"
AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_users() -> dict:
    if AUTH_FILE.exists():
        try:
            return json.loads(AUTH_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_users(users: dict):
    AUTH_FILE.write_text(json.dumps(users, indent=2))


def signup(name: str, email: str, password: str) -> dict:
    """Register a new user."""
    users = _load_users()
    email = email.lower().strip()

    if email in users:
        return {"success": False, "error": "Email already registered."}

    if len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters."}

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[email] = {
        "name":       name,
        "email":      email,
        "password":   hashed,
        "created_at": datetime.now().isoformat(),
        "collection": f"user_{hashlib.md5(email.encode()).hexdigest()[:8]}",
    }
    _save_users(users)
    logger.info(f"  New user registered: {email}")
    return {"success": True, "user": users[email]}


def login(email: str, password: str) -> dict:
    """Authenticate a user."""
    users = _load_users()
    email = email.lower().strip()

    if email not in users:
        return {"success": False, "error": "Email not found."}

    user = users[email]
    if not bcrypt.checkpw(password.encode(), user["password"].encode()):
        return {"success": False, "error": "Incorrect password."}

    return {"success": True, "user": user}


def get_user_collection(email: str) -> str:
    """Get the ChromaDB collection name for a user."""
    users = _load_users()
    email = email.lower().strip()
    if email in users:
        return users[email]["collection"]
    return f"user_{hashlib.md5(email.encode()).hexdigest()[:8]}"
