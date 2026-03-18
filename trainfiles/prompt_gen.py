import os
from openai import OpenAI


file_path = 'system.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    systemcontent = file.read()


client = OpenAI(
    api_key="api_key",  # Key[citation:3]
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # [citation:3]
)


completion = client.chat.completions.create(
    model="qwen-plus",  # "qwen-max", "qwen3-max"[citation:3]
    messages=[
        {"role": "system", "content": systemcontent},
        {"role": "user", "content":  "An artist paints a landscape on a canvas, surrounded by scattered brushes, a palette with vibrant colors, and a vase of sunflowers."}
    ],
    temperature=0.1 
)
print(completion.choices[0].message.content)