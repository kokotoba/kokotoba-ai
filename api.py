"""FastAPIアプリケーションのエントリーポイント。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.card_suggestion_service import CardSuggestionService
from core.chat_manager import ChatManager
from database.card_suggestion_repository import CardSuggestionRepository
from database.init import DatabaseManager
from router.card_suggestions import router as card_suggestions_router


def create_app(
    card_suggestion_service: CardSuggestionService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if card_suggestion_service is not None:
            app.state.card_suggestion_service = card_suggestion_service
            yield
            return

        from llm import LiteRTLMClient

        db_manager = DatabaseManager()
        llm_client = LiteRTLMClient(
            "models/gemma-4-E2B-it.litertlm",
            enable_speculative_decoding=True,
        )
        llm_client.start()
        chat_manager = ChatManager(db_manager, llm_client)
        chat_manager.warm_up()
        app.state.card_suggestion_service = CardSuggestionService(
            chat_manager,
            CardSuggestionRepository(db_manager),
        )
        try:
            yield
        finally:
            llm_client.close()

    application = FastAPI(
        title="Kokotoba API",
        version="1.0.0",
        lifespan=lifespan,
    )
    if card_suggestion_service is not None:
        application.state.card_suggestion_service = card_suggestion_service
    application.include_router(card_suggestions_router)
    return application


app = create_app()
