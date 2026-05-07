from pipeline import chat
from tools import tools
from functions import available_functions


prompts = [
    "请帮我查询合肥的天气",
    "请一句话介绍下你自己",
    "请帮我查询一下数据库中pk库的dept表有多少条数据",
]

for prompt in prompts:
    print(f"\n\n用户提问：{prompt}")
    result = chat(
        user_prompt=prompt,
        tools=tools,
        available_functions=available_functions,
        verbose=True
    )
    print(f"最终回答：{result}")
    print("-" * 50)
