import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

client = OpenAI(api_key="test", base_url=os.getenv("OLLAMA_BASE_URL"))
chat = client.chat.completions.create(
         model="qwen3.5:4b",
         messages=[{"role": "user", "content": "请介绍你自己"}]
)


print(chat.choices[0].message.content)
