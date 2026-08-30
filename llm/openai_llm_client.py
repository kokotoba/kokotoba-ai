"""OpenAI Responses API を利用する LLMClient 実装。"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from llm.llm_client import LLMClient


class OpenAILLMClient(LLMClient):
    """発話カードを Structured Outputs で生成するOpenAIクライアント。"""

    DEFAULT_MODEL = "gpt-5.6-luna"
    CARD_RESPONSE_FORMAT = {
        "type": "json_schema",
        "name": "card_suggestions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "cards": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
            "required": ["cards"],
            "additionalProperties": False,
        },
    }

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        client: Any | None = None,
    ) -> None:
        self._model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._timeout_seconds = timeout_seconds or float(
            os.getenv("OPENAI_TIMEOUT_SECONDS", "30")
        )
        self._client = client
        self._injected_client = client is not None

    def start(self) -> None:
        """SDKクライアントを初期化する。ここではAPI通信を行わない。"""
        if self._client is not None:
            return
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when using OpenAILLMClient"
            )

        options: dict[str, Any] = {
            "api_key": self._api_key,
            "timeout": self._timeout_seconds,
        }
        if self._base_url:
            options["base_url"] = self._base_url
        self._client = OpenAI(**options)

    def warm_up_connection(self) -> None:
        """生成を行わず、OpenAI APIへの接続を事前に確立する。"""
        if self._client is None:
            raise RuntimeError(
                "OpenAILLMClient is not started. Call start() before warm-up."
            )

        # モデル情報の取得は推論トークンを使わず、以降のResponses API呼び出しと
        # 同じSDKクライアントのHTTP接続プールを温められる。
        self._client.models.retrieve(self._model)

    def generate(
        self,
        prompt: str,
        max_output_tokens: int | None = None,
    ) -> str:
        """低レイテンシ設定とJSON Schemaを使ってカード候補を生成する。"""
        if not prompt.strip():
            raise ValueError("prompt must not be empty or whitespace only")
        if self._client is None:
            raise RuntimeError(
                "OpenAILLMClient is not started. Call start() before generate()."
            )

        request: dict[str, Any] = {
            "model": self._model,
            "input": prompt,
            "reasoning": {"effort": "none"},
            "text": {
                "format": self.CARD_RESPONSE_FORMAT,
                "verbosity": "low",
            },
            # 会話内容をモデル改善用に保存しない。
            "store": False,
        }
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens

        response = self._client.responses.create(**request)
        output_text = response.output_text
        if not isinstance(output_text, str) or not output_text.strip():
            raise RuntimeError("OpenAI Responses API returned no output text")
        return output_text

    def close(self) -> None:
        """SDKのHTTP接続を解放する。注入されたテスト用クライアントは触らない。"""
        client = self._client
        self._client = None
        if client is None or self._injected_client:
            return

        close_method = getattr(client, "close", None)
        if callable(close_method):
            close_method()
