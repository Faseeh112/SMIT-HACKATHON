"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def temp_db(tmp_path):
    """A throwaway SQLite DB for tests that touch the database layer."""
    from database import db

    db_path = tmp_path / "test.db"
    db.init_db(db_path=db_path)

    from config.settings import settings

    original = settings.sqlite_db_path
    object.__setattr__(settings, "sqlite_db_path", db_path)
    yield db
    object.__setattr__(settings, "sqlite_db_path", original)
