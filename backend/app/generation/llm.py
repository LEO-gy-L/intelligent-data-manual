from collections.abc import Iterator
from backend.app.config import DEEPSEEK_API_KEY, LLM_MODEL
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def get_llm(temperature: float = 0.7) -> ChatDeepSeek:
    """
    全项目唯一出口：返回配置好的 ChatDeepSeek 实例。
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY 未设置，请检查 .env 文件")
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        temperature=temperature,
    )


def ask(
    query: str | list[str] | list,
    system: str = "",
    stream: bool = False,
    batch: bool = False,
    temperature: float = 0.7,
) -> str | list[str] | Iterator:
    """
    包装器：支持单条、批量、流式、多轮历史四种调用方式。

    参数:
        query:
            - str          -> 单条提问
            - list[str]    -> 批量提问（需 batch=True）
            - list         -> 消息历史列表（多轮对话时用）
        system:   系统提示词，会放在每条对话的最前面
        stream:   True 则返回流式迭代器，需用 for 循环读取
        batch:    True 则把 list[str] 当批量请求并行发送
        temperature: 采样温度，0~2，越高越有创意

    返回:
        str | list[str] | Iterator
    """
    llm = get_llm(temperature=temperature)

    messages: list = []
    if system:
        messages.append(SystemMessage(content=system))

    # 1) 单条字符串
    if isinstance(query, str):
        messages.append(HumanMessage(content=query))
        if stream:
            return llm.stream(messages)
        response = llm.invoke(messages)
        return response.content

    # 2) 批量模式（List[str]）
    if batch and isinstance(query, list):
        batch_messages = []
        for q in query:
            item = [HumanMessage(content=q)]
            if system:
                item.insert(0, SystemMessage(content=system))
            batch_messages.append(item)
        responses = llm.batch(batch_messages)
        return [r.content for r in responses]

    # 3) 多轮历史模式（List[Message 对象]）
    if isinstance(query, list):
        if stream:
            return llm.stream(query)
        response = llm.invoke(query)
        return response.content

    raise TypeError(f"不支持的 query 类型: {type(query)}")
