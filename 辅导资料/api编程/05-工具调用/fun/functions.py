import json
import pymysql

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


available_functions = {
    "get_weather": get_weather,
    "send_msg": send_msg,
    "sql_query": sql_query
}