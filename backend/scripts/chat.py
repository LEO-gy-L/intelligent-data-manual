import sys
from pathlib import Path

# 把项目根目录加入 Python 路径，确保能 import backend
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.generation.llm import get_llm
from langchain_core.messages import HumanMessage, AIMessage


def chat_loop():
    llm = get_llm(temperature=0.7)
    history: list = []  # 里面存 HumanMessage / AIMessage 对象

    print("=" * 40)
    print("Datasheet RAG v2 - 多轮对话 CLI")
    print("命令：q=退出  c=清空历史")
    print("=" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue
        if user_input.lower() == "q":
            print("再见。")
            break
        if user_input.lower() == "c":
            history.clear()
            print("[历史已清空]")
            continue

        # 历史 + 新问题 一起丢给模型
        history.append(HumanMessage(content=user_input))
        response = llm.invoke(history)
        history.append(AIMessage(content=response.content))

        print(f"AI: {response.content}")


if __name__ == "__&#8203;main__":
    chat_loop()
