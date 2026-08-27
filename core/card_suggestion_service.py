"""発話カード生成APIのユースケース。"""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from core.chat_manager import ChatManager
from database.card_suggestion_repository import (
    CardSuggestion,
    CardSuggestionRepository,
    SuggestedCard,
)


class SuggestionNotFoundError(Exception):
    pass


class CardNotFoundError(Exception):
    pass


class SelectionConflictError(Exception):
    pass


class CardSuggestionService:
    """カード生成と選択記録を一つのAPI操作としてまとめる。"""

    def __init__(
        self,
        chat_manager: ChatManager,
        repository: CardSuggestionRepository,
    ) -> None:
        self._chat_manager = chat_manager
        self._repository = repository
        # LiteRTLMClientと選択記録は単一プロセス内で直列実行する。
        self._lock = Lock()

    def create_suggestion(
        self,
        question: str,
        location: str,
        latitude: float | None = None,
        longitude: float | None = None,
        fast: bool = True,
    ) -> CardSuggestion:
        with self._lock:
            question_type = self._chat_manager.classify_question(question)
            card_texts = self._chat_manager.handle_user_input(
                question,
                location,
                latitude=0.0 if latitude is None else latitude,
                longitude=0.0 if longitude is None else longitude,
                fast=fast,
            )

        suggestion = CardSuggestion(
            id=f"csg_{uuid4().hex}",
            question=question,
            location=location,
            question_type=question_type,
            generation_mode="fast" if fast else "quality",
            cards=[
                SuggestedCard(id=f"card_{uuid4().hex}", text=text)
                for text in card_texts
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._repository.create(suggestion)
        return suggestion

    def select_card(
        self,
        suggestion_id: str,
        card_id: str,
    ) -> tuple[CardSuggestion, SuggestedCard, str]:
        with self._lock:
            suggestion = self._repository.get(suggestion_id)
            if suggestion is None:
                raise SuggestionNotFoundError(suggestion_id)

            selected_card = next(
                (card for card in suggestion.cards if card.id == card_id),
                None,
            )
            if selected_card is None:
                raise CardNotFoundError(card_id)

            if suggestion.selected_card_id is not None:
                if suggestion.selected_card_id != card_id:
                    raise SelectionConflictError(suggestion_id)
                return suggestion, selected_card, suggestion.selected_at or ""

            self._chat_manager.record_selected_card(
                user_input=suggestion.question,
                user_location_input=suggestion.location,
                shown_cards=[card.text for card in suggestion.cards],
                selected_card=selected_card.text,
                generation_mode=suggestion.generation_mode,
            )
            selected_at = datetime.now(timezone.utc).isoformat()
            self._repository.record_selection(
                suggestion_id,
                card_id,
                selected_at,
            )
            return suggestion, selected_card, selected_at
