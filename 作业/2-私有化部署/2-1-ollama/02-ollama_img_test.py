
from openai import OpenAI

client = OpenAI(api_key="test", base_url="http://106.75.244.51:11434/v1")

import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "./test.jpg"
base64_image = encode_image(image_path)


response = client.chat.completions.create(
    model="qwen3-vl:8b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图的内容"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
    max_tokens=8000,
)
print(response.choices[0].message.content)