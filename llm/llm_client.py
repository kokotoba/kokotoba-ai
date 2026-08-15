"""LLM 実装に共通するインターフェース。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """上位処理が利用する、LLM 固有実装に依存しない抽象クラス。"""

    @abstractmethod
    def start(self) -> None:
        """モデルを読み込み、推論可能な状態にする。"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """プロンプトからテキストを生成して返す。"""

    @abstractmethod
    def close(self) -> None:
        """モデルおよび会話に関連するリソースを解放する。"""
