from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException
from pydantic import ValidationError

from core.card_suggestion_service import CardSuggestionService
from database.card_suggestion_repository import CardSuggestionRepository
from database.init import DatabaseManager
from router.card_suggestions import (
    CreateSuggestionRequest,
    SelectCardRequest,
    SuggestionContextRequest,
    create_suggestion,
    select_card,
)


class FakeChatManager:
    def __init__(self) -> None:
        self.selections: list[dict[str, object]] = []

    def classify_question(self, user_input: str) -> str:
        return "yes_no"

    def handle_user_input(
        self,
        user_input: str,
        user_location_input: str,
        latitude: float = 0.0,
        longitude: float = 0.0,
        fast: bool = False,
    ) -> list[str]:
        return ["はいお願いします", "いいえ大丈夫です"]

    def record_selected_card(self, **selection: object) -> None:
        self.selections.append(selection)


class CardSuggestionsApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        db_manager = DatabaseManager.__new__(DatabaseManager)
        db_manager.path = Path(self.temporary_directory.name) / "test.sqlite3"
        db_manager.initialize()

        self.chat_manager = FakeChatManager()
        self.service = CardSuggestionService(
            self.chat_manager,  # type: ignore[arg-type]
            CardSuggestionRepository(db_manager),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_suggestion(self):
        return create_suggestion(
            CreateSuggestionRequest(
                question="袋は必要ですか？",
                context=SuggestionContextRequest(place_name="コンビニ"),
            ),
            self.service,
        )

    def test_create_and_select_card(self) -> None:
        suggestion = self._create_suggestion()
        self.assertEqual(suggestion.question_type, "yes_no")
        self.assertEqual(len(suggestion.cards), 2)

        selected_card = suggestion.cards[0]
        selection_response = select_card(
            suggestion.id,
            SelectCardRequest(card_id=selected_card.id),
            self.service,
        )

        self.assertEqual(
            selection_response.selected_card,
            selected_card,
        )
        self.assertEqual(len(self.chat_manager.selections), 1)

        with self.assertRaises(HTTPException) as conflict_context:
            select_card(
                suggestion.id,
                SelectCardRequest(card_id=suggestion.cards[1].id),
                self.service,
            )
        self.assertEqual(conflict_context.exception.status_code, 409)

        # 同じ選択の再送は成功扱いにし、学習履歴は重複登録しない。
        select_card(
            suggestion.id,
            SelectCardRequest(card_id=selected_card.id),
            self.service,
        )
        self.assertEqual(len(self.chat_manager.selections), 1)

    def test_rejects_unknown_suggestion_and_card(self) -> None:
        with self.assertRaises(HTTPException) as missing_context:
            select_card(
                "csg_missing",
                SelectCardRequest(card_id="card_missing"),
                self.service,
            )
        self.assertEqual(missing_context.exception.status_code, 404)

        suggestion = self._create_suggestion()
        with self.assertRaises(HTTPException) as card_context:
            select_card(
                suggestion.id,
                SelectCardRequest(card_id="card_missing"),
                self.service,
            )
        self.assertEqual(card_context.exception.status_code, 422)

    def test_rejects_blank_inputs(self) -> None:
        with self.assertRaises(ValidationError):
            CreateSuggestionRequest(
                question="   ",
                context=SuggestionContextRequest(place_name="コンビニ"),
            )


if __name__ == "__main__":
    unittest.main()
