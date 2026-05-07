import os

from openai import OpenAI
from dotenv import load_dotenv

from chatbot import MemoryChatBot

load_dotenv(override=True)

client = OpenAI(
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

MODEL = "deepseek-chat"

bot = MemoryChatBot(
    client=client,
    model=MODEL,
    session_id="demo_001",
    store_dir="./sessions",
    max_history=4,  # 保留最近 4 轮对话（8 条消息）
    compress_min_messages=4,
    system_prompt="你是一个简洁的AI助手。请用1-2句话回答，不要展开。",
)

# 多轮对话，模拟足够长的会话以触发压缩
conversations = [
    "我是詹姆斯，从事大模型相关工作。",
    "你还记得我的名字和职业吗？",
    "我最近在研究 RAG 技术。",
    "RAG 里向量数据库用什么比较好？",
    "我比较倾向于用 Milvus，你觉得怎么样？",
    "另外我还想了解一下 Agent 的实现方式。",
    "ReAct 和 Function Calling 有什么区别？",
    "好的，我打算先从 Function Calling 开始实践。",
    "你能帮我总结一下我们刚才聊了哪些话题吗？",
]

for i, user_msg in enumerate(conversations):
    print(f"[轮次 {i+1}] 用户: {user_msg}")
    reply = bot.chat(user_msg)
    print(f"[轮次 {i+1}] 助手: {reply}\n")

# 最终状态汇总
print("=" * 60)
print("最终状态汇总")
print("=" * 60)

print(f"\n[保留的消息] ({len(bot.get_history())} 条):")
for i, msg in enumerate(bot.get_history()):
    print(f"  {i+1}. {msg['role']}: {msg['content']}")

summary = bot.get_summary()
if summary:
    print(f"\n[压缩摘要]:\n  {summary}")
else:
    print("\n[压缩摘要]: 未触发压缩（消息数不足）")
