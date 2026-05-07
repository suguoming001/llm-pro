from dataclasses import dataclass
from typing import Any


@dataclass
class RollingSummaryCompressor:
    client: Any
    model: str
    min_messages: int = 4
    keep_ratio: float = 0.5
    prefix: str = "[以下是之前对话的摘要]"

    def compress(self, session: Any) -> Any:
        messages = session.messages

        if len(messages) < self.min_messages:
            return session

        split_index = max(self.min_messages, int(len(messages) * self.keep_ratio))
        to_compress = messages[:split_index]
        to_keep = messages[split_index:]

        summary_input = self._build_summary_input(
            session.compress_content, to_compress
        )
        summary = self._summarize(summary_input)

        session.compress_content = summary
        session.messages = to_keep
        return session

    def _build_summary_input(
        self, existing_summary: str, messages: list[dict[str, str]]
    ) -> str:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        if existing_summary:
            return f"[之前摘要]\n{existing_summary}\n\n[新增对话]\n{history_text}"
        return history_text

    def _summarize(self, content: str) -> str:
        prompt = f"""
请将以下内容压缩成统一摘要，保留关键事实：
- 用户身份
- 技术背景
- 已完成任务
- 未完成任务
- 重要决策

要求：
- 使用第三人称（用户 / 助手）
- 100~200字
- 必须以「{self.prefix}」开头

内容：
{content}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
        return response.choices[0].message.content
