from datetime import datetime

from core.rag import RAG
from database.init import DatabaseManager
from llm.llm_client import LLMClient
from memory.memory_consolidator import (
    Location,
    MemoryConsolidator,
    ShortTermMemoryRecord,
)


class ChatManager:

    def __init__(
        self,
        db_manager: DatabaseManager,
        llm_client: LLMClient,
    ) -> None:
        self.db_manager = db_manager
        self.memory_consolidator = MemoryConsolidator(db_manager)
        self.rag = RAG(llm_client, db_manager)

    def handle_user_input(
        self,
        user_input: str,
        user_location_input: str,
    ) -> list[str]:
        # 短期記憶に会話内容を保存
        self.memory_consolidator.record_short_term_memory(
            ShortTermMemoryRecord(
                conversation_text=user_input,
                timestamp=datetime.now(),
                location=Location(
                    latitude=0.0,          # 必要に応じて実際の緯度
                    longitude=0.0,         # 必要に応じて実際の経度
                    place_name=user_location_input,
                ),
                speaker="user",
            )
        )

        return self.rag.generate_rag_response(
            user_input,
            user_location_input,
        )

    def record_selected_card(
        self,
        user_input: str,
        user_location_input: str,
        shown_cards: list[str],
        selected_card: str,
    ) -> None:
        """ユーザーが実際に選んだカードを記録する。"""
        self.rag.record_selected_card(
            user_input=user_input,
            user_location=user_location_input,
            shown_cards=shown_cards,
            selected_card=selected_card,
        )

    def finish_session(self) -> None:
        return

    def _build_card(self, user_input: str, user_location_input: str) -> str:
        return f"{user_input} / {user_location_input}"
