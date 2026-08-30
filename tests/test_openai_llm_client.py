from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm.openai_llm_client import OpenAILLMClient


class FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(output_text=self.output_text)


class FakeModels:
    def __init__(self) -> None:
        self.retrieved_model_ids: list[str] = []

    def retrieve(self, model: str) -> SimpleNamespace:
        self.retrieved_model_ids.append(model)
        return SimpleNamespace(id=model)


class FakeOpenAI:
    def __init__(self, output_text: str) -> None:
        self.responses = FakeResponses(output_text)
        self.models = FakeModels()


class OpenAILLMClientTest(unittest.TestCase):
    def test_requires_api_key_when_starting_real_client(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = OpenAILLMClient()

        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
            client.start()

    def test_generates_structured_card_response(self) -> None:
        fake_openai = FakeOpenAI('{"cards":["はいお願いします"]}')
        client = OpenAILLMClient(
            model="test-model",
            client=fake_openai,
        )

        client.start()
        result = client.generate("候補を作って", max_output_tokens=96)

        self.assertEqual(result, '{"cards":["はいお願いします"]}')
        request = fake_openai.responses.requests[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["reasoning"], {"effort": "none"})
        self.assertEqual(request["max_output_tokens"], 96)
        self.assertEqual(request["store"], False)
        self.assertEqual(
            request["text"],
            {
                "format": OpenAILLMClient.CARD_RESPONSE_FORMAT,
                "verbosity": "low",
            },
        )

    def test_warms_up_connection_without_generating(self) -> None:
        fake_openai = FakeOpenAI("")
        client = OpenAILLMClient(model="test-model", client=fake_openai)

        client.start()
        client.warm_up_connection()

        self.assertEqual(fake_openai.models.retrieved_model_ids, ["test-model"])
        self.assertEqual(fake_openai.responses.requests, [])

    def test_warm_up_requires_started_client(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            client = OpenAILLMClient()

        with self.assertRaisesRegex(RuntimeError, "not started"):
            client.warm_up_connection()

    def test_rejects_empty_output(self) -> None:
        client = OpenAILLMClient(client=FakeOpenAI(""))

        with self.assertRaisesRegex(RuntimeError, "no output text"):
            client.generate("候補を作って")


if __name__ == "__main__":
    unittest.main()
