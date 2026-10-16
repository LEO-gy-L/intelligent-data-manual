import sys
from pathlib import Path

# 把项目根目录加入 Python 路径，确保能 import backend
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.generation.llm import get_llm
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

MAX_MESSAGES = 10  # 只发最近 10 条消息给模型（约 5 轮对话）

def chat_loop():
    llm = get_llm(temperature=0.7)
    history: list[BaseMessage] = []  # 里面存 HumanMessage / AIMessage 对象

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

        cmd = user_input.lower()
        if cmd == "q":
            print("再见。")
            break
        if cmd == "c":
            history.clear()
            print("[历史已清空]")
            continue

        # 历史 + 新问题 一起丢给模型
       
        history.append(HumanMessage(content=user_input))
        # 内存保护：历史太长时自动丢弃旧消息，只保留最近 20 条
        if len(history) > MAX_MESSAGES * 2:
            history = history[-MAX_MESSAGES * 2:]
        try:
            print("AI: ", end="", flush=True)
            chunks = []
            for chunk in llm.stream(history[-MAX_MESSAGES:]):
                print(chunk.content, end="", flush=True)
                chunks.append(chunk.content)
            print()  # 最后换行
        except Exception as e:
            print(f"\n[调用失败：{e}]")
            history.pop()
            continue
        history.append(AIMessage(content="".join(chunks)))
if __name__ == "__main__":
    chat_loop()