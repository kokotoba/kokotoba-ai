# kokotoba-ai

## Database

このサービスは `kokotoba-infra` の PostgreSQL を利用します。先に infra の
`up.sh` でDB起動と Flyway マイグレーションを完了してください。

```sh
cd ../kokotoba-infra
cp .env.example .env
bash ./up.sh

cd ../kokotoba-ai
export DATABASE_URL=postgresql://kokotoba:kokotoba_dev_password@localhost:5432/kokotoba
poetry run uvicorn api:app
```

テーブル定義をサービス側へ追加してはいけません。スキーマ変更は
`kokotoba-infra/postgres/migrations` に新しい Flyway SQL を追加します。

既存の SQLite データを初回だけ移す場合は、infra のマイグレーション後に実行します。
同じIDはスキップされるため再実行できます。

```sh
poetry run python -m scripts.migrate_sqlite_to_postgres data/app.sqlite3
```
