from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from database.init import DatabaseManager


@dataclass
class Location:
    latitude: float
    longitude: float
    place_name: str


@dataclass
class ShortTermMemoryRecord:
    conversation_text: str
    timestamp: datetime
    location: Optional[Location]
    speaker: str


class MemoryConsolidator:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

        # セッションごとに終了するので短期記憶はテスト段階では、メモリに保存する
        self.short_term_memory: list[ShortTermMemoryRecord] = []

    def record_short_term_memory(self, record: ShortTermMemoryRecord) -> None:
        self.short_term_memory.append(record)
