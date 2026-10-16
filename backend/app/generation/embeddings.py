from backend.app.config import DASHSCOPE_API_KEY, EMBED_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    批量把字符串列表转成向量列表。
    使用阿里云 DashScope 官方 SDK。
    """
    import dashscope

    if not DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY 未设置，请检查 .env 文件")

    dashscope.api_key = DASHSCOPE_API_KEY

    # DashScope embedding 接口
    from dashscope import TextEmbedding

    resp = TextEmbedding.call(
        model=EMBED_MODEL,  # "text-embedding-v4"
        input=texts,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Embedding 调用失败：{resp.message}")

    # resp.output["embeddings"] 是 [{embedding: [...], text_index: 0}, ...]
    items = sorted(resp.output["embeddings"], key=lambda x: x["text_index"])
    return [item["embedding"] for item in items]



def embed_query(text: str) -> list[float]:
    """
    把单个查询字符串转成向量。
    """
    return embed_texts([text])[0]
