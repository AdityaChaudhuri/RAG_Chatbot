"""Postgres connection helper."""
from contextlib import contextmanager

from .config import settings


@contextmanager
def connect():
    import psycopg
    from pgvector.psycopg import register_vector

    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)  # lets us pass numpy arrays straight into vector columns
        yield conn
