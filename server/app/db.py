"""Database session and engine.

Postgres is the target (`DATABASE_URL`), but the default is a SQLite file so
the prototype runs on a laptop with nothing installed. Nothing in the models
uses a Postgres-only type, so the two stay interchangeable.

    DATABASE_URL=postgresql+psycopg://user:pw@localhost/roadsurvey
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

SERVER_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SERVER_DIR / "data"
MEDIA_DIR = SERVER_DIR / "media"

DEFAULT_URL = f"sqlite:///{(DATA_DIR / 'roadsurvey.db').as_posix()}"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_URL)

_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    # SQLite guards against cross-thread use by default; FastAPI's threadpool
    # legitimately hands sessions between threads.
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=not _is_sqlite,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency -- one session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Connection string with any password redacted, for the startup banner."""
    if "@" in DATABASE_URL and "//" in DATABASE_URL:
        scheme, rest = DATABASE_URL.split("//", 1)
        creds, host = rest.split("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}//{user}:***@{host}"
    return DATABASE_URL
