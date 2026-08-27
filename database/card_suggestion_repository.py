"""APIで生成した発話カード候補の永続化。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from psycopg.types.json import Jsonb

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
    generation_mode: str
    cards: list[SuggestedCard]
    created_at: str
    selected_card_id: str | None = None
    selected_at: str | None = None


class CardSuggestionRepository:
    """生成結果を保存し、後続の選択APIから参照できるようにする。"""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db_manager = db_manager

    def create(self, suggestion: CardSuggestion) -> None:
        cards_json = [
            {"id": card.id, "text": card.text} for card in suggestion.cards
        ]
        with self._db_manager.connect() as conn:
            conn.execute(
                """
                INSERT INTO card_suggestions (
                    id, question, location, question_type, generation_mode,
                    cards, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    suggestion.id,
                    suggestion.question,
                    suggestion.location,
                    suggestion.question_type,
                    suggestion.generation_mode,
                    Jsonb(cards_json),
                    suggestion.created_at,
                ),
            )
            conn.commit()

    def get(self, suggestion_id: str) -> CardSuggestion | None:
        with self._db_manager.connect() as conn:
            row = conn.execute(
                """
                SELECT id, question, location, question_type,
                       generation_mode, cards, created_at,
                       selected_card_id, selected_at
                FROM card_suggestions
                WHERE id = %s
                """,
                (suggestion_id,),
            ).fetchone()

        if row is None:
            return None

        cards = [
            SuggestedCard(id=value["id"], text=value["text"])
            for value in row["cards"]
        ]
        return CardSuggestion(
            id=row["id"],
            question=row["question"],
            location=row["location"],
            question_type=row["question_type"],
            generation_mode=row["generation_mode"],
            cards=cards,
            created_at=self._timestamp(row["created_at"]),
            selected_card_id=row["selected_card_id"],
            selected_at=self._timestamp(row["selected_at"]),
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
                SET selected_card_id = %s, selected_at = %s
                WHERE id = %s
                """,
                (card_id, selected_at, suggestion_id),
            )
            conn.commit()

    @staticmethod
    def _timestamp(value: datetime | str | None) -> str | None:
        if isinstance(value, datetime):
            return value.isoformat()
        return value
