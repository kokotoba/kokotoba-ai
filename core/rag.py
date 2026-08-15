"""長期記憶と選択履歴を利用して発話カードを生成するRAG処理。"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Literal

import numpy as np

from database.card_selection_history import CardSelectionHistory
from database.long_term_memory_searcher import LongTermMemorySearcher
from llm.llm_client import LLMClient
from util.embedding import Embedding

if TYPE_CHECKING:
    from database.init import DatabaseManager

QuestionType = Literal["yes_no", "choice", "open"]


class RAG:
    """質問に対して、ユーザーが選択できる短い発話カードを生成する。"""

    MAX_CARDS = 5
    MAX_CARD_LENGTH = 24

    def __init__(
        self,
        llm_client: LLMClient,
        db_manager: DatabaseManager,
    ) -> None:
        self.llm_client = llm_client
        self.db_manager = db_manager
        self.embedding = Embedding()
        self.long_term_memory_searcher = LongTermMemorySearcher(db_manager)
        self.card_selection_history = CardSelectionHistory(db_manager)

    def generate_rag_response(
        self,
        user_input: str,
        user_location: str,
    ) -> list[str]:
        """質問と文脈から、検証済みの発話カード候補を返す。"""
        question_type = self._classify_question(user_input)
        base_cards = self._base_cards(question_type, user_input)

        expanded_query = self._query_expansion(user_input, user_location)
        query_embedding = self.embedding.embed(expanded_query)
        long_term_memory_result = self.long_term_memory_searcher.search(
            query_embedding,
            minimum_similarity=0.84,
        )

        interaction_embedding = self._embed_interaction(
            user_input,
            user_location,
        )
        previously_selected_cards = self.card_selection_history.find_relevant(
            interaction_embedding,
            user_location,
        )

        generated_cards = self._generate_cards_with_llm(
            user_input=user_input,
            user_location=user_location,
            question_type=question_type,
            base_cards=base_cards,
            previously_selected_cards=previously_selected_cards,
            long_term_memory_result=long_term_memory_result,
        )
        cards = self._merge_cards(
            question_type=question_type,
            base_cards=base_cards,
            generated_cards=generated_cards,
            previously_selected_cards=previously_selected_cards,
        )

        card_candidates = " ".join(cards)
        print(f"カード候補: {card_candidates}")
        return cards

    def record_selected_card(
        self,
        user_input: str,
        user_location: str,
        shown_cards: list[str],
        selected_card: str,
    ) -> None:
        """実際に選択されたカードを次回の候補順位へ反映できるよう保存する。"""
        interaction_embedding = self._embed_interaction(
            user_input,
            user_location,
        )
        self.card_selection_history.record(
            question=user_input,
            location=user_location,
            shown_cards=shown_cards,
            selected_card=selected_card,
            question_embedding=interaction_embedding,
        )

    def _query_expansion(self, user_input: str, user_location: str) -> str:
        """元の質問と場所を維持しつつ、長期記憶検索用の関連語を補う。"""
        if not user_input.strip():
            raise ValueError("user_input must not be empty or whitespace only")
        if not user_location.strip():
            raise ValueError("user_location must not be empty or whitespace only")

        prompt = f"""
あなたは、ユーザーの長期記憶をベクトル検索するためのクエリ拡張器です。
次の質問と現在地から、関連する過去の記録にヒットしそうな日本語の関連語を
最大5個生成してください。

<question>
{user_input}
</question>

<location>
{user_location}
</location>

ルール:
- 質問への回答はしない
- 現在地を特定の施設種別に決めつけない
- 行動、出来事、話題、言い換えを中心に補う
- 具体的な体験、症状、商品、人物、日時を捏造しない
- 関連語だけを半角スペース区切りで出力する
- 見出し、番号、説明、引用符、句読点、改行は出力しない
""".strip()

        expanded_terms = " ".join(self.llm_client.generate(prompt).split())
        query_parts = [user_input.strip(), user_location.strip()]
        if expanded_terms:
            query_parts.append(expanded_terms)
        return " ".join(query_parts)

    def _generate_cards_with_llm(
        self,
        *,
        user_input: str,
        user_location: str,
        question_type: QuestionType,
        base_cards: list[str],
        previously_selected_cards: list[str],
        long_term_memory_result: str,
    ) -> list[str]:
        """LLMにJSON形式の追加候補を生成させ、検証して返す。"""
        prompt = f"""
あなたは、発語障害のあるユーザー向けの発話カードを作る支援者です。
ユーザーがボタンを押すだけで相手の質問へ答えられるように、短く直接的な
一人称の回答候補を作ってください。

<question>
{user_input}
</question>

<location>
{user_location}
</location>

<question_type>
{question_type}
</question_type>

<excluded_base_cards>
{json.dumps(base_cards, ensure_ascii=False)}
</excluded_base_cards>

<previously_selected_cards>
{json.dumps(previously_selected_cards, ensure_ascii=False)}
</previously_selected_cards>

<long_term_memory>
{long_term_memory_result or "関連度が十分な長期記憶なし"}
</long_term_memory>

ルール:
- excluded_base_cardsは別処理で追加するため、絶対に出力しない
- excluded_base_cardsより具体的で、状況に合った追加候補だけを1〜3個作る
- 過去に同様の状況で選ばれたカードは有力候補として扱う
- 長期記憶は参考情報であり、現在も同じ状態だと断定しない
- 関連度が十分な長期記憶に症状、希望、行動が含まれる場合は、その内容を
  ユーザーが選べる候補として1〜2個追加してよい
- 長期記憶と無関係な症状、希望、体験、商品などを捏造しない
- 医療診断や確定していない事実を作らない
- 関連する長期記憶も具体的な追加候補もない場合はcardsを空配列にする
- 各候補は24文字以内にする
- 説明、Markdown、コードフェンスは出力しない
- 必ず次のJSON形式だけを出力する

{{"cards":["候補1","候補2","候補3"]}}
""".strip()

        response = self.llm_client.generate(prompt)
        cards = self._parse_cards_json(response)
        return self._filter_generated_cards(
            cards,
            question_type=question_type,
            user_input=user_input,
        )

    def _classify_question(self, user_input: str) -> QuestionType:
        """質問文をルールベースで回答形式に分類する。"""
        normalized = user_input.strip()
        if not normalized:
            raise ValueError("user_input must not be empty or whitespace only")

        choice_markers = ("どちら", "どっち", "どれ", "それとも", "または")
        if any(marker in normalized for marker in choice_markers):
            return "choice"

        open_markers = (
            "どう",
            "いかが",
            "何",
            "なに",
            "いつ",
            "どこ",
            "誰",
            "だれ",
            "なぜ",
            "どの",
        )
        if any(marker in normalized for marker in open_markers):
            return "open"

        if normalized.endswith(("?", "？", "か", "か。")):
            return "yes_no"
        return "open"

    def _base_cards(
        self,
        question_type: QuestionType,
        user_input: str,
    ) -> list[str]:
        """質問形式ごとに、必ず利用できる安全な基本候補を返す。"""
        if question_type == "yes_no":
            if "温め" in user_input or "あたため" in user_input:
                return [
                    "はいお願いします",
                    "いいえそのままで",
                    "少しだけお願いします",
                    "もう一度お願いします",
                ]
            if "袋" in user_input:
                return [
                    "はいお願いします",
                    "いいえ大丈夫です",
                    "袋を分けてください",
                    "もう一度お願いします",
                ]
            return [
                "はいお願いします",
                "いいえ大丈夫です",
                "わかりません",
                "もう一度お願いします",
            ]

        if question_type == "choice":
            return [
                "最初のもの",
                "次のもの",
                "どちらでもいいです",
                "もう一度お願いします",
            ]

        if "体調" in user_input or "具合" in user_input:
            return [
                "元気です",
                "少し具合が悪いです",
                "わかりません",
                "うまく説明できません",
            ]
        return [
            "わかりません",
            "うまく説明できません",
            "もう一度お願いします",
        ]

    def _parse_cards_json(self, response: str) -> list[str]:
        """LLMレスポンスからJSONを抽出し、安全なカード文字列だけを返す。"""
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if match is None:
            return []

        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        raw_cards = payload.get("cards") if isinstance(payload, dict) else None
        if not isinstance(raw_cards, list):
            return []

        cards: list[str] = []
        for value in raw_cards:
            if not isinstance(value, str):
                continue
            card = re.sub(
                r"[\s、。,．・:：\"'`「」『』【】\[\]]+",
                "",
                value,
            )
            if not card or len(card) > self.MAX_CARD_LENGTH:
                continue
            if card not in cards:
                cards.append(card)
        return cards[: self.MAX_CARDS]

    def _merge_cards(
        self,
        *,
        question_type: QuestionType,
        base_cards: list[str],
        generated_cards: list[str],
        previously_selected_cards: list[str],
    ) -> list[str]:
        """履歴・固定候補・LLM候補を優先順位付きで統合する。"""
        if question_type == "yes_no":
            ordered_cards = (
                previously_selected_cards
                + base_cards[:3]
                + generated_cards
                + base_cards[3:]
            )
        else:
            ordered_cards = (
                previously_selected_cards
                + generated_cards
                + base_cards
            )

        cards: list[str] = []
        for card in ordered_cards:
            normalized = re.sub(r"\s+", "", card)
            if not normalized or len(normalized) > self.MAX_CARD_LENGTH:
                continue
            if normalized not in cards:
                cards.append(normalized)
            if len(cards) == self.MAX_CARDS:
                break
        return cards

    @staticmethod
    def _filter_generated_cards(
        cards: list[str],
        *,
        question_type: QuestionType,
        user_input: str,
    ) -> list[str]:
        """質問へ直接答えていない過去形の候補などを除外する。"""
        if question_type == "yes_no":
            generic_yes_no_cards = {
                "はい",
                "いいえ",
                "はいお願いします",
                "いいえ大丈夫です",
                "大丈夫です",
                "結構です",
            }
            cards = [
                card for card in cards if card not in generic_yes_no_cards
            ]

        if question_type == "open" and (
            "体調" in user_input or "具合" in user_input
        ):
            past_action_markers = (
                "ました",
                "だった",
                "行った",
                "もらった",
                "言われた",
            )
            return [
                card
                for card in cards
                if not any(marker in card for marker in past_action_markers)
            ]
        return cards

    def _embed_interaction(
        self,
        user_input: str,
        user_location: str,
    ) -> np.ndarray:
        """質問と場所をカード選択履歴検索用のベクトルへ変換する。"""
        return self.embedding.embed(
            f"質問: {user_input.strip()} 場所: {user_location.strip()}"
        )
