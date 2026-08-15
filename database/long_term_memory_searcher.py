"""長期記憶のベクトル検索処理。"""

from __future__ import annotations

import sqlite3

import numpy as np

from database.init import DatabaseManager


class LongTermMemorySearcher:
    """DatabaseManagerを利用して長期記憶を類似度検索する。"""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db_manager = db_manager

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 3,
        minimum_similarity: float = 0.80,
    ) -> str:
        """長期記憶をコサイン類似度で検索し、上位結果を文字列で返す。"""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between -1.0 and 1.0")

        query_vector = np.asarray(query_embedding, dtype=np.float32)
        if query_vector.ndim != 1 or query_vector.size == 0:
            raise ValueError("query_embedding must be a non-empty 1D vector")

        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0.0:
            raise ValueError("query_embedding must not be a zero vector")

        with self._db_manager.connect() as conn:
            rows = conn.execute("""
                SELECT
                    id,
                    summary,
                    source_text,
                    embedding,
                    place_name,
                    speaker,
                    event_time
                FROM long_term_memory
                WHERE embedding IS NOT NULL
            """).fetchall()

        scored_rows: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            memory_vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if memory_vector.shape != query_vector.shape:
                continue

            memory_norm = float(np.linalg.norm(memory_vector))
            if memory_norm == 0.0:
                continue

            similarity = float(
                np.dot(query_vector, memory_vector)
                / (query_norm * memory_norm)
            )
            if similarity < minimum_similarity:
                continue
            scored_rows.append((similarity, row))

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        top_rows: list[tuple[float, sqlite3.Row]] = []
        seen_source_texts: set[str] = set()
        for similarity, row in scored_rows:
            source_text = row["source_text"]
            if source_text in seen_source_texts:
                continue
            seen_source_texts.add(source_text)
            top_rows.append((similarity, row))
            if len(top_rows) == limit:
                break

        if not top_rows:
            return ""

        results: list[str] = []
        for index, (similarity, row) in enumerate(top_rows, start=1):
            details = [
                f"検索結果{index}（類似度: {similarity:.4f}）",
                f"要約: {row['summary']}",
                f"内容: {row['source_text']}",
            ]
            if row["place_name"]:
                details.append(f"場所: {row['place_name']}")
            if row["speaker"]:
                details.append(f"話者: {row['speaker']}")
            if row["event_time"]:
                details.append(f"日時: {row['event_time']}")
            results.append("\n".join(details))

        return "\n\n".join(results)
