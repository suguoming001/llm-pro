from dotenv import load_dotenv
import os
from openai import OpenAI
import pymysql

load_dotenv(override=True)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
) 
MODEL_NAME = "gpt-5-nano"


def get_weather(city):
    return {
        "城市": city,
        "天气": "晴",
        "温度": "25°C",
        "湿度": "60%",
        "风速": "5 km/h"
    } 

def send_msg(message):
    """
    发送天气提醒给用户
    """
    
    return {
        "status": "success",
        "message": f"已发送消息: {message}"
    }


def sql_query(sql_statement):
    """
    查询本地MySQL数据库，执行一段SQL代码，并返回查询结果
    """
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='000000',
        database='pk',
        charset='utf8'
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_statement)
            result = cursor.fetchall()
    finally:
        connection.close()
    
    return json.dumps(result, ensure_ascii=True)  # 将查询结果转换成JSON字符串格式返回


   

# 将工具函数封装成符合规范的工具描述
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气信息，包括：天气、温度、湿度和风速等；当用户询问天气时，应该调用这个工具；",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "要查询的城市名称，例如：北京",
                    }
                },
                "required": ["city"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_msg",
            "description": "发送天气提醒给用户",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "要发送的消息内容",
                    }
                },
                "required": ["message"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "查询本地MySQL数据库，执行一段SQL代码，并返回查询结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_statement": {
                        "type": "string",
                        "description": "字符串形式的SQL查询语句，用于执行对MySQL数据库中pk库进行查询",
                    }
                },
                "required": ["sql_statement"]
            },
        }
    },
]


available_functions = {
    "get_weather": get_weather,
    "send_msg": send_msg,
    "sql_query": sql_query
}

import json
def append_function_messages(messages, response):
    # 拼接第一次大模型请求的返回结果
    messages.append(response.choices[0].message.model_dump())
    tool_calls = response.choices[0].message.tool_calls
    

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print("="*20)
        print(f"工具调用: {tool_name}, 参数: {tool_args}")

        function_to_call = available_functions[tool_name]  # 获取到对应的工具函数

        try:
            function_response = function_to_call(**tool_args)
            print(f"工具调用: {tool_name}, 执行结果: {function_response}")
            messages.append({
                "role": "tool",   # 固定写法，表示这是工具调用的结果
                "tool_call_id": tool_call.id,   # 关联到具体的工具调用id
                "content": str(function_response)  # 工具执行的结果内容，必须是字符串格式
            })
            
        except Exception as e:
            function_response = f"工具执行出错: {str(e)}"   
        
    return messages


def chat_with_tools(messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        parallel_tool_calls=True
    )
    
    if response.choices[0].finish_reason == "tool_calls":
        while True:
            messages = append_function_messages(messages, response)
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                parallel_tool_calls=True
            )
            if response.choices[0].finish_reason != "tool_calls":
                break
    return response

# messages = [
#     {"role": "user", "content": "请帮我查询一下北京和深圳的天气，并将查询结果发送给詹姆斯" }
# ]
# 
messages = [
    {"role": "user", "content": "求emp表中每个部门的人数，部门列为deptno" }
]  

print(chat_with_tools(messages).choices[0].message.content)

# 查数据库这种需求：一定要有字段或者表的描述信息

