from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(override=True)

# 统一使用openai sdk
def test_deepseek():
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )  # 编程入口点

    response = client.chat.completions.create(
        model="deepseek-chat",  # 指定你要调用的模型名称
        messages=[
            {"role": "user", "content": "请介绍一下你自己"}  # role=user :  你向大模型发起的提问
        ]
    )
    print(response.choices[0].message.content)


def test_mimo():
    client = OpenAI(
        api_key=os.getenv("MIMO_API_KEY"),
        base_url=os.getenv("MIMO_BASE_URL")
    )  # 编程入口点

    response = client.chat.completions.create(
        model="mimo-v2-pro",  # 指定你要调用的模型名称
        messages=[
            {"role": "user", "content": "请介绍一下你自己"}  # role=user :  你向大模型发起的提问
        ]
    )
    print(response.choices[0].message.content)

def test_openai():
    client = OpenAI()  # 编程入口点

    response = client.chat.completions.create(
        model="gpt-4.1-nano",  # 指定你要调用的模型名称
        messages=[
            {"role": "user", "content": "请介绍一下你自己"}  # role=user :  你向大模型发起的提问
        ]
    )
    print(response.choices[0].message.content)

def test_openrouter():
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL")
    )  # 编程入口点

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",  # 指定你要调用的模型名称
        messages=[
            {"role": "user", "content": "请介绍一下你自己"}  # role=user :  你向大模型发起的提问
        ]
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    test_deepseek()
    # test_mimo()
    # test_openai()
    # test_openrouter()