"""Database connection and table management for NeuroFlow session persistence."""

import os
import sqlite3
from pathlib import Path
from neuroflow.config import get_settings


def get_db_path() -> Path:
    """Return the absolute path to the SQLite persistence database."""
    settings = get_settings()
    db_path = Path(settings.sqlite_db_path)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parent.parent.parent.parent / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def init_db(db_path: Path | str | None = None) -> None:
    """Initialize SQLite database tables if they do not exist."""
    path = Path(db_path) if db_path else get_db_path()
    if path != Path(":memory:"):
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            query TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            model_provider TEXT,
            model_name TEXT,
            plan_json TEXT,
            approval_json TEXT,
            execution_records_json TEXT,
            results_json TEXT,
            final_report_json TEXT,
            metadata_json TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open and return a connection to the SQLite database."""
    path = Path(db_path) if db_path else get_db_path()
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
