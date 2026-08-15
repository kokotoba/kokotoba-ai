"""APIで生成した発話カード候補の永続化。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from database.init import DatabaseManager


@dataclass(frozen=True)
class SuggestedCard:
    id: str
    text: str


@dataclass(frozen=True)
class CardSuggestion:
    id: str
    question: str
    location: str
    question_type: str
    cards: list[SuggestedCard]
    created_at: str
    selected_card_id: str | None = None
    selected_at: str | None = None


class CardSuggestionRepository:
    """生成結果を保存し、後続の選択APIから参照できるようにする。"""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db_manager = db_manager
        self._initialize()

    def create(self, suggestion: CardSuggestion) -> None:
        cards_json = json.dumps(
            [{"id": card.id, "text": card.text} for card in suggestion.cards],
            ensure_ascii=False,
        )
        with self._db_manager.connect() as conn:
            conn.execute(
                """
                INSERT INTO card_suggestions (
                    id, question, location, question_type, cards, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    suggestion.id,
                    suggestion.question,
                    suggestion.location,
                    suggestion.question_type,
                    cards_json,
                    suggestion.created_at,
                ),
            )
            conn.commit()

    def get(self, suggestion_id: str) -> CardSuggestion | None:
        with self._db_manager.connect() as conn:
            row = conn.execute(
                """
                SELECT id, question, location, question_type, cards,
                       created_at, selected_card_id, selected_at
                FROM card_suggestions
                WHERE id = ?
                """,
                (suggestion_id,),
            ).fetchone()

        if row is None:
            return None

        cards = [
            SuggestedCard(id=value["id"], text=value["text"])
            for value in json.loads(row["cards"])
        ]
        return CardSuggestion(
            id=row["id"],
            question=row["question"],
            location=row["location"],
            question_type=row["question_type"],
            cards=cards,
            created_at=row["created_at"],
            selected_card_id=row["selected_card_id"],
            selected_at=row["selected_at"],
        )

    def record_selection(
        self,
        suggestion_id: str,
        card_id: str,
        selected_at: str,
    ) -> None:
        with self._db_manager.connect() as conn:
            conn.execute(
                """
                UPDATE card_suggestions
                SET selected_card_id = ?, selected_at = ?
                WHERE id = ?
                """,
                (card_id, selected_at, suggestion_id),
            )
            conn.commit()

    def _initialize(self) -> None:
        with self._db_manager.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS card_suggestions (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    location TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    cards TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    selected_card_id TEXT,
                    selected_at DATETIME
                )
                """
            )
            conn.commit()
