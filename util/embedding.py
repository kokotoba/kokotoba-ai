"""アプリケーション内で共通利用するテキストEmbedding処理。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedding:
    """入力文字列をmultilingual-e5-smallでEmbeddingする汎用クラス。"""

    DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"
    DEFAULT_MODEL_PATH = Path(
        "models/sentence-transformers/multilingual-e5-small"
    )

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        """モデル情報を保持する。実際の読み込みは最初のEmbedding時に行う。"""
        self._model_id = model_id
        self._model_path = Path(model_path)
        self._model: SentenceTransformer | None = None

    def embed(
        self,
        text: str,
    ) -> np.ndarray:
        """1件の文字列をfloat32のEmbeddingベクトルへ変換する。"""
        if not text.strip():
            raise ValueError("text must not be empty or whitespace only")

        # multilingual-e5では検索用の入力に "query: " を付与する。
        embedding = self._get_model().encode(f"query: {text}")
        return np.asarray(embedding, dtype=np.float32)

    def _get_model(self) -> SentenceTransformer:
        """モデルをインスタンス内で一度だけ読み込む。"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> SentenceTransformer:
        """モデルを初回だけ取得し、以降はローカルから読み込む。"""
        completion_marker = self._model_path / ".download_complete"

        if completion_marker.is_file():
            print(f"保存済みのEmbeddingモデルを読み込んでいます: {self._model_path}")
            return SentenceTransformer(
                str(self._model_path),
                local_files_only=True,
            )

        print(f"Embeddingモデルを初回ダウンロードしています: {self._model_id}")
        model = SentenceTransformer(self._model_id)

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(self._model_path))
        completion_marker.touch()
        print(f"Embeddingモデルを保存しました: {self._model_path}")
        return model
