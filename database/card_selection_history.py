"""ユーザーが実際に選択した発話カードの保存と検索。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import numpy as np

from database.init import DatabaseManager


class CardSelectionHistory:
    """質問と場所に対して選ばれたカードを永続化して再利用する。"""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db_manager = db_manager
        self._initialize()

    def record(
        self,
        question: str,
        location: str,
        shown_cards: list[str],
        selected_card: str,
        question_embedding: np.ndarray,
    ) -> None:
        """ユーザーが選択したカードと、そのときの文脈を保存する。"""
        if not question.strip():
            raise ValueError("question must not be empty or whitespace only")
        if not location.strip():
            raise ValueError("location must not be empty or whitespace only")
        if selected_card not in shown_cards:
            raise ValueError("selected_card must be included in shown_cards")

        embedding = self._validate_embedding(question_embedding)
        with self._db_manager.connect() as conn:
            conn.execute(
                """
                INSERT INTO card_selection_history (
                    question,
                    location,
                    shown_cards,
                    selected_card,
                    question_embedding,
                    selected_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    question,
                    location,
                    json.dumps(shown_cards, ensure_ascii=False),
                    selected_card,
                    embedding.tobytes(),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            conn.commit()

    def find_relevant(
        self,
        question_embedding: np.ndarray,
        location: str,
        limit: int = 3,
        minimum_similarity: float = 0.85,
    ) -> list[str]:
        """現在の質問に近い履歴から、過去に選ばれたカードを返す。"""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between -1.0 and 1.0")

        query_vector = self._validate_embedding(question_embedding)
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm == 0.0:
            raise ValueError("question_embedding must not be a zero vector")

        with self._db_manager.connect() as conn:
            rows = conn.execute(
                """
                SELECT location, selected_card, question_embedding, selected_at
                FROM card_selection_history
                WHERE question_embedding IS NOT NULL
                ORDER BY selected_at DESC
                """
            ).fetchall()

        scored_rows: list[tuple[bool, float, sqlite3.Row]] = []
        for row in rows:
            history_vector = np.frombuffer(
                row["question_embedding"],
                dtype=np.float32,
            )
            if history_vector.shape != query_vector.shape:
                continue

            history_norm = float(np.linalg.norm(history_vector))
            if history_norm == 0.0:
                continue

            similarity = float(
                np.dot(query_vector, history_vector)
                / (query_norm * history_norm)
            )
            if similarity < minimum_similarity:
                continue

            same_location = row["location"].strip() == location.strip()
            scored_rows.append((same_location, similarity, row))

        scored_rows.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        selected_cards: list[str] = []
        for _, _, row in scored_rows:
            card = row["selected_card"]
            if card not in selected_cards:
                selected_cards.append(card)
            if len(selected_cards) == limit:
                break
        return selected_cards

    def _initialize(self) -> None:
        """カード選択履歴テーブルを作成する。"""
        with self._db_manager.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS card_selection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    location TEXT NOT NULL,
                    shown_cards TEXT NOT NULL,
                    selected_card TEXT NOT NULL,
                    question_embedding BLOB NOT NULL,
                    selected_at DATETIME NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _validate_embedding(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding must be a non-empty 1D vector")
        return vector
