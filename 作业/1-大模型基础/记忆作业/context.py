from dataclasses import dataclass
from typing import Any


@dataclass
class ContextWindowBuilder:
    max_history: int = 10  # 保留最近 N 轮对话（1轮 = user + assistant）
    system_prompt: str = "你是一个有用的AI助手。"

    def build(self, session: Any) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []

        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})

        if session.compress_content:
            msgs.append({"role": "system", "content": session.compress_content})

        recent_messages = session.messages[-self.max_history * 2:]
        msgs.extend(recent_messages)
        return msgs
