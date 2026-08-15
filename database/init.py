import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# Embedding計算用に追加
import numpy as np
from sentence_transformers import SentenceTransformer


class DatabaseManager:
    EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
    EMBEDDING_MODEL_PATH = Path(
        "models/sentence-transformers/multilingual-e5-small"
    )

    def __init__(self):
        self.path = Path("data/app.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row

        # SQLiteではデフォルトで無効なので有効化
        conn.execute("PRAGMA foreign_keys = ON")

        # sqlite-vecをロード
        conn.enable_load_extension(True)

        import sqlite_vec
        sqlite_vec.load(conn)

        conn.enable_load_extension(False)

        return conn

    def initialize(self):
        with self.connect() as conn:
            # 長期記憶 (embedding BLOB を追加)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                summary TEXT NOT NULL,
                source_text TEXT NOT NULL,
                embedding BLOB,  -- 追加: ベクトルデータを保存するカラム

                place_name TEXT,
                latitude REAL,
                longitude REAL,

                speaker TEXT,
                event_time DATETIME,

                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                modified_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # タグ辞書
            conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                name TEXT NOT NULL UNIQUE,
                description TEXT
            )
            """)

            # 長期記憶とタグの対応
            conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_tag_map (
                memory_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,

                PRIMARY KEY (memory_id, tag_id),

                FOREIGN KEY (memory_id)
                    REFERENCES long_term_memory(id)
                    ON DELETE CASCADE,

                FOREIGN KEY (tag_id)
                    REFERENCES tags(id)
                    ON DELETE CASCADE
            )
            """)
            conn.commit()

    def insert_demo_data(self):
        # 完全に独立したユニークなエピソードデータのリスト
        demo_records = [
            {
                "summary": "風邪の診察と薬の処方",
                "source_text": "昨晩から熱が38度出たため、九州病院を受診した。医師の診察の結果、ただの風邪とのことで解熱剤と抗生物質を3日分処方された。しっかり休むように言われた。",
                "place_name": "九州病院",
                "latitude": 33.8631,
                "longitude": 130.7606,
                "speaker": "自分",
                "days_ago": 2,
                "tags": ["hospital", "health", "medicine", "sick"]
            },
            {
                "summary": "新規プロジェクトのキックオフ会議",
                "source_text": "東京ビッグサイトの会議室で、来期に向けたAI開発プロジェクトのキックオフミーティングを行った。各部署から合計20名が集まり、今後のロードマップについて熱心な議論が交わされた。",
                "place_name": "東京ビッグサイト",
                "latitude": 35.6298,
                "longitude": 139.7938,
                "speaker": "山田マネージャー",
                "days_ago": 10,
                "tags": ["work", "meeting", "ai", "project"]
            },
            {
                "summary": "自己ベスト更新のランニング",
                "source_text": "早朝に大濠公園へ行き、5kmのジョギングをした。気温が涼しく走りやすかったため、ペースが上がり、自己ベストのタイムを2分更新することができた。とても気分が良い。",
                "place_name": "大濠公園",
                "latitude": 33.5858,
                "longitude": 130.3763,
                "speaker": "自分",
                "days_ago": 5,
                "tags": ["sports", "running", "hobby", "achievement"]
            },
            {
                "summary": "新幹線の大幅遅延トラブル",
                "source_text": "出張帰りに京都駅から新幹線に乗ろうとしたところ、静岡県での大雨の影響で運転が見合わせられていた。駅構内で3時間待機することになり、非常に疲労した。",
                "place_name": "京都駅",
                "latitude": 34.9858,
                "longitude": 135.7587,
                "speaker": "駅アナウンス",
                "days_ago": 20,
                "tags": ["travel", "trouble", "train"]
            },
            {
                "summary": "最新のノートパソコンを購入",
                "source_text": "これまで使っていたPCのバッテリーが寿命を迎えたため、秋葉原のヨドバシカメラで最新のM3チップ搭載のノートパソコンを購入した。想定より安く買えたので、浮いたお金でマウスも新調した。",
                "place_name": "ヨドバシカメラ マルチメディアAkiba",
                "latitude": 35.6987,
                "longitude": 139.7747,
                "speaker": "店員さん",
                "days_ago": 1,
                "tags": ["shopping", "gadget", "pc"]
            },
            {
                "summary": "友人との海沿いドライブとランチ",
                "source_text": "大学時代の友人3人と車で糸島半島へドライブに行った。海沿いのカフェで食べたガーリックシュリンプが絶品で、夕日も見ることができ素晴らしい休日になった。",
                "place_name": "糸島のカフェ",
                "latitude": 33.6269,
                "longitude": 130.1583,
                "speaker": "友人A",
                "days_ago": 45,
                "tags": ["drive", "food", "friends", "holiday"]
            },
            {
                "summary": "引っ越しに伴う転入届の提出",
                "source_text": "先週末に新居への引っ越しが完了したため、午前休を取って区役所の窓口に行き、転入届とマイナンバーカードの住所変更手続きを行った。非常に混雑していた。",
                "place_name": "福岡市中央区役所",
                "latitude": 33.5881,
                "longitude": 130.3956,
                "speaker": "窓口担当者",
                "days_ago": 60,
                "tags": ["procedure", "moving", "government"]
            },
            {
                "summary": "深夜の地震による一時避難",
                "source_text": "深夜2時頃に震度4の地震が発生。スマホの緊急地震速報で飛び起きた。念のため防災リュックを持ってマンションのロビーまで一時避難したが、特に被害はなく30分後に部屋に戻った。",
                "place_name": "自宅マンション",
                "latitude": None,
                "longitude": None,
                "speaker": "自分",
                "days_ago": 120,
                "tags": ["emergency", "earthquake", "disaster"]
            },
            {
                "summary": "図書館での資格試験の勉強",
                "source_text": "来月受験予定の基本情報技術者試験に向けて、市立図書館の学習室で過去問題集を解いた。アルゴリズムの問題がまだ苦手なので、重点的に復習する必要がある。",
                "place_name": "市立図書館",
                "latitude": 33.5899,
                "longitude": 130.3541,
                "speaker": "自分",
                "days_ago": 15,
                "tags": ["study", "qualification", "library"]
            }
        ]

        # アプリ起動のたびに同じデモデータが増えないようにする。
        with self.connect() as conn:
            existing_source_texts = {
                row["source_text"]
                for row in conn.execute(
                    "SELECT source_text FROM long_term_memory"
                ).fetchall()
            }
        demo_records = [
            record
            for record in demo_records
            if record["source_text"] not in existing_source_texts
        ]
        if not demo_records:
            print("デモデータは既に登録されています。")
            return

        model = self._load_embedding_model()

        # E5モデルの要件：保存するドキュメントには "passage: " を付ける
        texts_to_embed = ["passage: " + r["source_text"] for r in demo_records]

        print("Embeddingを計算しています...")
        # encode にリストを渡すことで、一括（バッチ）で高速にベクトル化されます
        embeddings = model.encode(texts_to_embed)

        with self.connect() as conn:
            cursor = conn.cursor()

            # データと計算済みのベクトルを同時にループ処理
            for record, emb in zip(demo_records, embeddings):
                # event_time の計算
                event_time = (datetime.now(
                ) - timedelta(days=record["days_ago"])).strftime("%Y-%m-%d %H:%M:%S")

                # ベクトル（numpy配列）を float32 のバイナリデータに変換
                emb_bytes = emb.astype(np.float32).tobytes()

                # 1. long_term_memory の挿入 (embedding を追加)
                cursor.execute("""
                        INSERT INTO long_term_memory
                        (summary, source_text, embedding, place_name, latitude, longitude, speaker, event_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                    record["summary"],
                    record["source_text"],
                    emb_bytes,           # 計算したバイナリデータをセット
                    record.get("place_name"),
                    record.get("latitude"),
                    record.get("longitude"),
                    record.get("speaker"),
                    event_time
                ))

                memory_id = cursor.lastrowid

                # 2. タグの処理とマッピング
                for tag_name in record["tags"]:
                    cursor.execute(
                        "INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
                    cursor.execute(
                        "SELECT id FROM tags WHERE name = ?", (tag_name,))
                    tag_row = cursor.fetchone()
                    tag_id = tag_row["id"]

                    # 3. memory_tag_map の挿入
                    cursor.execute("""
                            INSERT INTO memory_tag_map (memory_id, tag_id)
                            VALUES (?, ?)
                        """, (memory_id, tag_id))

            # トランザクションのコミット
            conn.commit()
            print(f"{len(demo_records)}件のユニークなデモデータを挿入し、Embeddingを保存しました。")

    @classmethod
    def _load_embedding_model(cls) -> SentenceTransformer:
        """Embeddingモデルを初回だけ取得し、以降はローカルから読み込む。"""
        completion_marker = cls.EMBEDDING_MODEL_PATH / ".download_complete"

        if completion_marker.is_file():
            print(
                "保存済みのEmbeddingモデルを読み込んでいます: "
                f"{cls.EMBEDDING_MODEL_PATH}"
            )
            return SentenceTransformer(
                str(cls.EMBEDDING_MODEL_PATH),
                local_files_only=True,
            )

        print(
            "Embeddingモデルを初回ダウンロードしています: "
            f"{cls.EMBEDDING_MODEL_ID}"
        )
        model = SentenceTransformer(cls.EMBEDDING_MODEL_ID)

        cls.EMBEDDING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(cls.EMBEDDING_MODEL_PATH))
        completion_marker.touch()
        print(f"Embeddingモデルを保存しました: {cls.EMBEDDING_MODEL_PATH}")
        return model
