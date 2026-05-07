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

print(f"✅ 已注册{len(tools)} 个工具：{[tool['function']['name'] for tool in tools]}")