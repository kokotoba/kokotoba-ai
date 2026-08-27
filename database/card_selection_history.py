"""ユーザーが実際に選択した発話カードの保存と検索。"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from psycopg.types.json import Jsonb

from database.init import DatabaseManager


class CardSelectionHistory:
    """質問と場所に対して選ばれたカードを永続化して再利用する。"""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db_manager = db_manager

    def record(
        self,
        question: str,
        location: str,
        shown_cards: list[str],
        selected_card: str,
        question_embedding: np.ndarray,
        generation_mode: str,
    ) -> None:
        """ユーザーが選択したカードと、そのときの文脈を保存する。"""
        if not question.strip():
            raise ValueError("question must not be empty or whitespace only")
        if not location.strip():
            raise ValueError("location must not be empty or whitespace only")
        if selected_card not in shown_cards:
            raise ValueError("selected_card must be included in shown_cards")
        if generation_mode not in ("fast", "quality"):
            raise ValueError("generation_mode must be fast or quality")

        embedding = self._db_manager.vector_literal(
            self._validate_embedding(question_embedding)
        )
        with self._db_manager.connect() as conn:
            conn.execute(
                """
                INSERT INTO card_selection_history (
                    question,
                    location,
                    shown_cards,
                    selected_card,
                    question_embedding,
                    selected_at,
                    generation_mode
                ) VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                """,
                (
                    question,
                    location,
                    Jsonb(shown_cards),
                    selected_card,
                    embedding,
                    datetime.now().astimezone(),
                    generation_mode,
                ),
            )
            conn.commit()

    def find_relevant(
        self,
        question_embedding: np.ndarray,
        location: str,
        generation_mode: str,
        limit: int = 3,
        minimum_similarity: float = 0.85,
        same_location_only: bool = False,
    ) -> list[str]:
        """現在の質問に近い履歴から、過去に選ばれたカードを返す。"""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between -1.0 and 1.0")
        if generation_mode not in ("fast", "quality"):
            raise ValueError("generation_mode must be fast or quality")

        query_vector = self._validate_embedding(question_embedding)
        if float(np.linalg.norm(query_vector)) == 0.0:
            raise ValueError("question_embedding must not be a zero vector")

        vector = self._db_manager.vector_literal(query_vector)
        with self._db_manager.connect() as conn:
            rows = conn.execute(
                """
                WITH query AS (SELECT %s::vector AS embedding)
                SELECT
                    location,
                    selected_card,
                    1 - (question_embedding <=> query.embedding) AS similarity
                FROM card_selection_history, query
                WHERE 1 - (question_embedding <=> query.embedding) >= %s
                  AND generation_mode = %s
                  AND (%s = FALSE OR BTRIM(location) = BTRIM(%s))
                ORDER BY
                    (BTRIM(location) = BTRIM(%s)) DESC,
                    question_embedding <=> query.embedding,
                    selected_at DESC
                """,
                (
                    vector,
                    minimum_similarity,
                    generation_mode,
                    same_location_only,
                    location,
                    location,
                ),
            ).fetchall()

        selected_cards: list[str] = []
        for row in rows:
            card = row["selected_card"]
            if card not in selected_cards:
                selected_cards.append(card)
            if len(selected_cards) == limit:
                break
        return selected_cards

    @staticmethod
    def _validate_embedding(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding must be a non-empty 1D vector")
        return vector
