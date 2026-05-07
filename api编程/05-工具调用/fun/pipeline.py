from dotenv import load_dotenv
import os
from openai import OpenAI
import json
load_dotenv(override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
) 

def chat(
    user_prompt: str,
    tools: list,
    available_functions: dict,
    system_prompt: str = "你是一个乐于助人的智能回答小助手。",
    model: str = "gpt-5-nano",
    verbose: bool = True
) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=model, 
        messages=messages,
        tools=tools
    )
    assistant_message = response.choices[0].message

    if not assistant_message.tool_calls:
        if verbose:
            print("LLM直接回复（不需要调用工具）")
        return assistant_message.content

    if verbose:
        print(f"LLM决定调用{len(assistant_message.tool_calls)}个工具")

    messages.append(assistant_message.model_dump())

    for tool_call in assistant_message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        if verbose:
            print(f"  → 准备调用工具：{func_name}({func_args})")

        if func_name in available_functions:
            function_result = available_functions[func_name](**func_args)
        else:
            function_result = {"error": f"未知工具：{func_name}"}

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(function_result)
        })

    final_response = client.chat.completions.create(
        model=model, 
        messages=messages,
        tools=tools, 
    )

    final_answer = final_response.choices[0].message.content
    if verbose:
        print(f"✅ 最终回答生成完成")

    return final_answer