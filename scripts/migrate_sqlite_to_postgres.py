"""既存の SQLite データを共有 PostgreSQL へ一度だけ取り込む。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
from psycopg.types.json import Jsonb

from database.init import DatabaseManager


def migrate(sqlite_path: Path, database_url: str | None) -> None:
    source = sqlite3.connect(sqlite_path)
    source.row_factory = sqlite3.Row
    target_manager = DatabaseManager(database_url)

    try:
        with target_manager.connect() as target:
            for row in source.execute("SELECT * FROM long_term_memory"):
                embedding = None
                if row["embedding"] is not None:
                    embedding = target_manager.vector_literal(
                        np.frombuffer(row["embedding"], dtype=np.float32)
                    )
                target.execute(
                    """
                    INSERT INTO long_term_memory (
                        id, summary, source_text, embedding, place_name,
                        latitude, longitude, speaker, event_time,
                        created_at, modified_at
                    ) OVERRIDING SYSTEM VALUE
                    VALUES (
                        %s, %s, %s, %s::vector, %s,
                        %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        row["id"],
                        row["summary"],
                        row["source_text"],
                        embedding,
                        row["place_name"],
                        row["latitude"],
                        row["longitude"],
                        row["speaker"],
                        row["event_time"],
                        row["created_at"],
                        row["modified_at"],
                    ),
                )

            for row in source.execute("SELECT * FROM tags"):
                target.execute(
                    """
                    INSERT INTO tags (id, name, description)
                    OVERRIDING SYSTEM VALUE VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    tuple(row),
                )

            for row in source.execute("SELECT * FROM memory_tag_map"):
                target.execute(
                    """
                    INSERT INTO memory_tag_map (memory_id, tag_id)
                    VALUES (%s, %s) ON CONFLICT DO NOTHING
                    """,
                    tuple(row),
                )

            for row in source.execute("SELECT * FROM card_suggestions"):
                target.execute(
                    """
                    INSERT INTO card_suggestions (
                        id, question, location, question_type, cards,
                        created_at, selected_card_id, selected_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        row["id"],
                        row["question"],
                        row["location"],
                        row["question_type"],
                        Jsonb(json.loads(row["cards"])),
                        row["created_at"],
                        row["selected_card_id"],
                        row["selected_at"],
                    ),
                )

            for row in source.execute("SELECT * FROM card_selection_history"):
                embedding = target_manager.vector_literal(
                    np.frombuffer(row["question_embedding"], dtype=np.float32)
                )
                target.execute(
                    """
                    INSERT INTO card_selection_history (
                        id, question, location, shown_cards, selected_card,
                        question_embedding, selected_at
                    ) OVERRIDING SYSTEM VALUE
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        row["id"],
                        row["question"],
                        row["location"],
                        Jsonb(json.loads(row["shown_cards"])),
                        row["selected_card"],
                        embedding,
                        row["selected_at"],
                    ),
                )

            for table in ("long_term_memory", "tags", "card_selection_history"):
                target.execute(
                    f"""
                    SELECT setval(
                        pg_get_serial_sequence('{table}', 'id'),
                        COALESCE((SELECT MAX(id) FROM {table}), 1),
                        (SELECT COUNT(*) > 0 FROM {table})
                    )
                    """
                )
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sqlite_path",
        nargs="?",
        type=Path,
        default=Path("data/app.sqlite3"),
    )
    parser.add_argument("--database-url")
    args = parser.parse_args()

    if not args.sqlite_path.is_file():
        parser.error(f"SQLite file not found: {args.sqlite_path}")
    migrate(args.sqlite_path, args.database_url)
    print("SQLite data migration completed.")


if __name__ == "__main__":
    main()
