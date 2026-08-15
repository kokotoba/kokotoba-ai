"""長期記憶のベクトル検索処理。"""

from __future__ import annotations

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

        if float(np.linalg.norm(query_vector)) == 0.0:
            raise ValueError("query_embedding must not be a zero vector")

        vector = self._db_manager.vector_literal(query_vector)
        with self._db_manager.connect() as conn:
            rows = conn.execute("""
                WITH query AS (SELECT %s::vector AS embedding),
                ranked AS (
                    SELECT
                        id,
                        summary,
                        source_text,
                        place_name,
                        speaker,
                        event_time,
                        1 - (embedding <=> query.embedding) AS similarity,
                        ROW_NUMBER() OVER (
                            PARTITION BY source_text
                            ORDER BY embedding <=> query.embedding
                        ) AS source_rank
                    FROM long_term_memory, query
                    WHERE embedding IS NOT NULL
                      AND 1 - (embedding <=> query.embedding) >= %s
                )
                SELECT *
                FROM ranked
                WHERE source_rank = 1
                ORDER BY similarity DESC
                LIMIT %s
            """, (vector, minimum_similarity, limit)).fetchall()

        if not rows:
            return ""

        results: list[str] = []
        for index, row in enumerate(rows, start=1):
            details = [
                f"検索結果{index}（類似度: {row['similarity']:.4f}）",
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
