from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from loguru import logger
from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session

from ytscribe.db.models import Base
from ytscribe.settings import get_settings


def register_sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@lru_cache
def get_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_settings().db_url
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    if url.startswith("sqlite"):
        register_sqlite_pragmas(engine)
    return engine


def add_missing_columns(engine: Engine) -> None:
    """Tiny forward-only migration: ALTER TABLE ADD COLUMN for model columns an older database lacks."""
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in have:
                    ddl = col.type.compile(engine.dialect)
                    logger.info("migrating: {}.{} {}", table.name, col.name, ddl)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ddl}'))


def init_db(engine: Engine | None = None) -> None:
    """Create all tables (idempotent) and add columns introduced since the database was created."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    add_missing_columns(engine)


@contextmanager
def get_session(engine: Engine | None = None) -> Iterator[Session]:
    with Session(engine or get_engine()) as s:
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
