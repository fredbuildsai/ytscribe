import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ytscribe.db.models import Base
from ytscribe.db.session import register_sqlite_pragmas


@pytest.fixture
def engine(tmp_path):
    e = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    register_sqlite_pragmas(e)
    Base.metadata.create_all(e)
    return e


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s
