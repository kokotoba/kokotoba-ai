"""LLMクライアントの公開インターフェース。"""

from typing import Any

from llm.llm_client import LLMClient

__all__ = ["LLMClient", "LiteRTLMClient", "OpenAILLMClient"]


def __getattr__(name: str) -> Any:
    """LiteRTへの依存を実際に利用する時点まで遅延させる。"""
    if name == "LiteRTLMClient":
        from llm.litert_lm_client import LiteRTLMClient

        return LiteRTLMClient
    if name == "OpenAILLMClient":
        from llm.openai_llm_client import OpenAILLMClient

        return OpenAILLMClient
    raise AttributeError(name)
