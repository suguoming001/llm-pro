
from openai import OpenAI

client = OpenAI(api_key="test", base_url="http://106.75.244.51:11434/v1")
chat = client.chat.completions.create(
         model="qwen3.5:4b",
         messages=[{"role": "user", "content": "请介绍你自己"}]
)


print(chat.choices[0].message.content)
