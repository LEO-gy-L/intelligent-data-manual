import os
from dotenv import load_dotenv
load_dotenv('.env')
key = os.getenv('DASHSCOPE_API_KEY')
print(f'key存在: {bool(key)}')
print(f'key前10个字符: {key[:10] if key else "空!"}')

import dashscope
dashscope.api_key = key
from dashscope import TextEmbedding
print('正在请求阿里云...')
resp = TextEmbedding.call(model='text-embedding-v4', input=['CPU是什么'])
print(f'状态码: {resp.status_code}')
print(f'embedding维度: {len(resp.output["embeddings"][0]["embedding"])}')
