# auth_db.py
import sqlite3
import os
import hashlib
import binascii
from typing import Tuple, Optional

# Use persistent dir (on Render this will be /persistent; locally we use ./sample_data)
PERSISTENT_DIR = os.getenv("PERSISTENT_DIR", os.path.join(os.path.dirname(__file__), "sample_data"))
os.makedirs(PERSISTENT_DIR, exist_ok=True)
DB_PATH = os.path.join(PERSISTENT_DIR, "users.db")

SALT_BYTES = 16
ITERATIONS = 100_000
HASH_NAME = "sha256"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password_hash TEXT,
            salt TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

def _hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    if salt is None:
        salt = os.urandom(SALT_BYTES)
    pwd = password.encode("utf-8")
    dk = hashlib.pbkdf2_hmac(HASH_NAME, pwd, salt, ITERATIONS)
    return binascii.hexlify(dk).decode(), binascii.hexlify(salt).decode()

def create_user(username: str, email: str, password: str) -> Tuple[bool, str]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? OR email=?", (username, email))
    if c.fetchone():
        conn.close()
        return False, "Username or email already exists"
    pwd_hash, salt_hex = _hash_password(password)
    try:
        c.execute(
            "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)",
            (username, email, pwd_hash, salt_hex),
        )
        conn.commit()
    except Exception as e:
        conn.close()
        return False, f"DB error: {e}"
    conn.close()
    return True, "User created"

def verify_user(username_or_email: str, password: str) -> Tuple[bool, Optional[str]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT username, password_hash, salt FROM users WHERE username=? OR email=?",
        (username_or_email, username_or_email),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return False, None
    username, stored_hash, salt_hex = row
    salt = binascii.unhexlify(salt_hex)
    computed_hash, _ = _hash_password(password, salt)
    if computed_hash == stored_hash:
        return True, username
    return False, None
