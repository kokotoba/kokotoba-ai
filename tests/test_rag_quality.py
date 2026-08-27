from __future__ import annotations

import unittest
from collections import OrderedDict

import numpy as np

from core.rag import RAG


class FakeLLMClient:
    def generate(
        self,
        prompt: str,
        max_output_tokens: int | None = None,
    ) -> str:
        if "風邪の診察と薬の処方" not in prompt:
            raise AssertionError("quality prompt must include retrieved memory")
        return '{"cards":["風邪と診断されました","薬を処方されました"]}'


class FakeEmbedding:
    def embed(self, text: str) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)


class FakeSelectionHistory:
    requested_mode: str | None = None

    def find_relevant(
        self,
        question_embedding: np.ndarray,
        location: str,
        generation_mode: str,
        same_location_only: bool = False,
    ) -> list[str]:
        self.requested_mode = generation_mode
        return ["元気です"]


class FakeLongTermMemorySearcher:
    def search(
        self,
        question_embedding: np.ndarray,
        minimum_similarity: float,
    ) -> str:
        return "要約: 風邪の診察と薬の処方"


class RagQualityTest(unittest.TestCase):
    def test_quality_prioritizes_rag_cards_over_quality_history(self) -> None:
        history = FakeSelectionHistory()
        rag = RAG.__new__(RAG)
        rag.llm_client = FakeLLMClient()
        rag.embedding = FakeEmbedding()
        rag.card_selection_history = history
        rag.long_term_memory_searcher = FakeLongTermMemorySearcher()
        rag._fast_cards_cache = OrderedDict()

        cards = rag.generate_rag_response(
            "最近九州病院ではなにがありましたか?",
            "場所情報なし",
            fast=False,
        )

        self.assertEqual(cards[0], "風邪と診断されました")
        self.assertEqual(cards[1], "薬を処方されました")
        self.assertEqual(history.requested_mode, "quality")


if __name__ == "__main__":
    unittest.main()
