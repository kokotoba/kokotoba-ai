from __future__ import annotations

import unittest
from types import TracebackType

import numpy as np

from database.long_term_memory_searcher import LongTermMemorySearcher


class FakeConnection:
    def __init__(self) -> None:
        self.sql = ""

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def execute(
        self,
        sql: str,
        parameters: tuple[str, float, int],
    ) -> FakeConnection:
        self.sql = sql
        return self

    def fetchall(self) -> list[object]:
        return []


class FakeDatabaseManager:
    def __init__(self) -> None:
        self.connection = FakeConnection()

    def vector_literal(self, embedding: np.ndarray) -> str:
        return "[1.0]"

    def connect(self) -> FakeConnection:
        return self.connection


class LongTermMemorySearcherTest(unittest.TestCase):
    def test_qualifies_embedding_column_in_vector_search(self) -> None:
        database = FakeDatabaseManager()
        searcher = LongTermMemorySearcher(database)  # type: ignore[arg-type]

        result = searcher.search(np.array([1.0], dtype=np.float32))

        self.assertEqual(result, "")
        self.assertIn(
            "memory.embedding <=> query.embedding",
            database.connection.sql,
        )
        self.assertIn(
            "FROM long_term_memory AS memory",
            database.connection.sql,
        )


if __name__ == "__main__":
    unittest.main()
