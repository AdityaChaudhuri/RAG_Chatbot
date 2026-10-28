"""Create the tables and indexes in the database named by DATABASE_URL.

Usage:
    python -m mr_summarizer.init_db

Needed for a cloud database such as Neon. (The Docker setup runs sql/schema.sql
automatically.) Safe to run more than once: every statement uses IF NOT EXISTS.
"""
from pathlib import Path

from .config import settings

SCHEMA = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


def main() -> None:
    import psycopg

    # plain connection: pgvector's type can't be registered until the extension exists
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(SCHEMA.read_text(encoding="utf-8"))
        conn.commit()
        n = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchone()[0]
    print(f"Schema ready ({n} tables).")


if __name__ == "__main__":
    main()
