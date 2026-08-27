from __future__ import annotations

import unittest
from collections import OrderedDict

import numpy as np

from core.rag import RAG


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        prompt: str,
        max_output_tokens: int | None = None,
    ) -> str:
        self.calls += 1
        return '{"cards":["海へ行きたいです","家で休みたいです","まだ未定です"]}'


class FailingEmbedding:
    def embed(self, text: str) -> np.ndarray:
        raise AssertionError("fast mode must not calculate an embedding")


class FakeSelectionHistory:
    def find_relevant(
        self,
        question_embedding: np.ndarray,
        location: str,
        generation_mode: str,
        same_location_only: bool = False,
    ) -> list[str]:
        raise AssertionError("fast mode must not read selection history")


class RagFastTest(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = FakeLLMClient()
        self.rag = RAG.__new__(RAG)
        self.rag.llm_client = self.llm
        self.rag.embedding = FailingEmbedding()
        self.rag.card_selection_history = FakeSelectionHistory()
        self.rag._fast_cards_cache = OrderedDict()

    def test_fast_mode_generates_generic_cards_and_caches_them(self) -> None:
        first = self.rag.generate_rag_response(
            "週末は何をしたいですか？",
            "海辺のホテル",
            fast=True,
        )
        second = self.rag.generate_rag_response(
            "週末は何をしたいですか？",
            "海辺のホテル",
            fast=True,
        )

        self.assertIn("海へ行きたいです", first)
        self.assertEqual(second, first)
        self.assertEqual(self.llm.calls, 1)


if __name__ == "__main__":
    unittest.main()
