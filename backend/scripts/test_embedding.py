import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.generation.embeddings import embed_texts, embed_query

if __name__ == "__main__":
    # 1. 批量测试
    texts = ["CPU是什么", "GPIO引脚定义"]
    vectors = embed_texts(texts)
    print(f"批量测试：生成了 {len(vectors)} 个向量")
    print(f"每个向量维度：{len(vectors[0])}")

    # 2. 单条测试
    query_vec = embed_query("什么是微控制器")
    print(f"单条测试：向量维度 {len(query_vec)}")

    # 3. 打印前 5 个维度看看数值
    print(f"第一个向量的前 5 维：{vectors[0][:5]}")
