"""
Database connection setup.

Location: src/admgpt/db/session.py

This file answers one question: "how do we talk to the database?"
It creates the engine, the session factory, the Base class for tables,
and a small helper (get_db) that every other db file uses.

Quick SQLAlchemy vocabulary for beginners:
- engine : the connection to the database file
- Session: a "workspace" for talking to the database (query, add, commit)
- Base   : the parent class of every table class (see models.py)
- commit : actually save the changes to the database file
"""

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from admgpt.config import DATA_DIR

logger = logging.getLogger(__name__)

# Build the database path from DATA_DIR (an ABSOLUTE path from config.py).
# The old version used the relative path "sqlite:///data/chatbot_memory.db",
# which created a different database depending on where you started the app.
DATABASE_URL = f"sqlite:///{DATA_DIR / 'chatbot_memory.db'}"

# The engine manages the actual connection to the database.
# check_same_thread=False: SQLite normally only allows the thread that created
# the connection to use it. Web servers use many threads, so we turn this check off.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# SessionLocal is a "factory": calling SessionLocal() gives you a new Session.
# autoflush=False      : do not automatically send pending changes before each query
# expire_on_commit=False: keep loaded values readable after commit(). Without this,
#     objects we return from get_db() would "expire" after the automatic commit and
#     raise an error when used after the session is closed.
# (Changes are never saved automatically; they are saved when get_db() commits.)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

# Every table class in models.py inherits from Base,
# so SQLAlchemy knows which classes are tables.
Base = declarative_base()


@contextmanager
def get_db():
    """
    Open a database session, and always clean it up afterwards.

    Use it with a `with` statement:

        with get_db() as db:
            db.add(some_object)      # change something
        # <- leaving the `with` block commits (saves) automatically

    What happens behind the scenes:
        1. Open a new session.
        2. Run your code inside the `with` block.
        3. No error -> commit (save the changes).
        4. Error    -> rollback (undo the changes) and re-raise the error.
        5. Always   -> close the session, so connections are never leaked.

    This replaces the old "try / commit / finally close" code that was
    copy-pasted into every function.

    Yields:
        Session: the database session to use inside the `with` block.
    """

    db = SessionLocal()

    try:
        yield db
        db.commit()  # save changes (does nothing harmful for read-only work)

    except Exception:
        db.rollback()  # something failed: undo any half-finished changes
        raise  # do not hide the error; let the caller see it

    finally:
        db.close()  # always release the session


def init_db():
    """
    Create all tables if they do not exist yet.

    Call this once when the app starts (for example in api/app.py).
    It is safe to call many times: existing tables are left untouched.
    """

    # Import models here so their classes are registered on Base BEFORE
    # create_all runs. (Importing at the top of this file would be circular,
    # because models.py itself imports Base from this file.)
    from admgpt.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables are ready: %s", DATABASE_URL)
