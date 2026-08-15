"""LiteRT-LM を利用する LLMClient の具象実装。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import litert_lm

from llm.llm_client import LLMClient


class LiteRTLMClient(LLMClient):
    """LiteRT-LM の Engine と Conversation を内部に隠蔽するクライアント。"""

    def __init__(self, model_path: str) -> None:
        """モデルファイルのパスを保持する。読み込みは start() で行う。"""
        self._model_path = Path(model_path).expanduser()
        self._engine: Any | None = None
        self._conversation: Any | None = None

    def start(self) -> None:
        """CPU のデフォルト設定でモデルと Conversation を初期化する。"""
        if self._engine is not None and self._conversation is not None:
            return

        # LiteRT-LM の詳細な INFO ログを非表示にする。
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

        # 以前の初期化が途中で失敗した場合に残ったリソースも解放する。
        if self._engine is not None or self._conversation is not None:
            self.close()

        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"LiteRT-LM model file was not found: {self._model_path}"
            )

        try:
            self._engine = litert_lm.Engine(str(self._model_path))
            self._conversation = self._engine.create_conversation()
        except Exception:
            self.close()
            raise

    def generate(self, prompt: str) -> str:
        """独立したConversationで同期生成し、テキストだけを返す。"""
        if not prompt.strip():
            raise ValueError("prompt must not be empty or whitespace only")

        self._ensure_started()
        conversation = self._conversation
        try:
            response = conversation.send_message(prompt)
            return self._extract_text(response)
        finally:
            # 内部プロンプトや前回の質問が次の推論へ混ざらないようにする。
            self._close_resource(conversation)
            self._conversation = None
            if self._engine is not None:
                self._conversation = self._engine.create_conversation()

    def close(self) -> None:
        """Conversation、Engine の順にリソースを安全に解放する。"""
        conversation = self._conversation
        engine = self._engine
        self._conversation = None
        self._engine = None

        try:
            self._close_resource(conversation)
        finally:
            self._close_resource(engine)

    def _ensure_started(self) -> None:
        """start() が正常に完了していることを確認する。"""
        if self._engine is None or self._conversation is None:
            raise RuntimeError(
                "LiteRTLMClient is not started. Call start() before generate()."
            )

    @staticmethod
    def _close_resource(resource: Any | None) -> None:
        """close() を持つリソースだけを解放する。"""
        if resource is None:
            return

        close_method = getattr(resource, "close", None)
        if callable(close_method):
            close_method()

    @staticmethod
    def _extract_text(response: Any) -> str:
        """LiteRT-LM のレスポンスからテキスト要素だけを抽出する。"""
        if isinstance(response, str):
            return response

        if isinstance(response, Mapping):
            direct_text = response.get("text")
            if isinstance(direct_text, str):
                return direct_text

            content = response.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, Sequence) and not isinstance(
                content, (str, bytes, bytearray)
            ):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, Mapping):
                        text = item.get("text")
                        item_type = item.get("type")
                        if isinstance(text, str) and item_type in (None, "text"):
                            text_parts.append(text)
                if text_parts:
                    return "".join(text_parts)

        # 将来のSDKが属性形式のレスポンスを返す場合にも最小限対応する。
        text_attribute = getattr(response, "text", None)
        if isinstance(text_attribute, str):
            return text_attribute

        raise TypeError(
            "Could not extract text from the LiteRT-LM response. "
            f"Unsupported response type: {type(response).__name__}"
        )
