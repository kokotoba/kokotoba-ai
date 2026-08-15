from core.chat_manager import ChatManager
from database.init import DatabaseManager
from llm import LLMClient, LiteRTLMClient


def main() -> None:
    db_manager = DatabaseManager()

    llm_manager: LLMClient = LiteRTLMClient(
        "models/gemma-4-E2B-it.litertlm"
    )

    try:
        llm_manager.start()
        chat_manager = ChatManager(db_manager, llm_manager)

        db_manager.insert_demo_data()
        print("デモデータの初期化が完了しました。")

        while True:
            print("talking start")
            user_input = input("Question: ").strip()
            if user_input.lower() == "exit":
                chat_manager.finish_session()
                break

            user_location_input = input("Your location: ").strip()
            cards = chat_manager.handle_user_input(
                user_input,
                user_location_input,
            )

            for index, card in enumerate(cards, start=1):
                print(f"{index}: {card}")

            selection = input(
                "選択したカード番号（保存しない場合はEnter）: "
            ).strip()
            if not selection:
                continue
            if not selection.isdigit() or not 1 <= int(selection) <= len(cards):
                print("有効なカード番号を入力してください。")
                continue

            selected_card = cards[int(selection) - 1]
            chat_manager.record_selected_card(
                user_input=user_input,
                user_location_input=user_location_input,
                shown_cards=cards,
                selected_card=selected_card,
            )
            print(f"選択を保存しました: {selected_card}")
    finally:
        llm_manager.close()


if __name__ == "__main__":
    main()
