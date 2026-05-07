from models import Session
from store import JsonSessionStore
from context import ContextWindowBuilder
from compressor import RollingSummaryCompressor


class MemoryChatBot:
    """整合持久化存储 + 上下文裁剪 + 滚动压缩的聊天机器人"""

    def __init__(
        self,
        client,
        model: str,
        session_id: str,
        store_dir: str = "./sessions",
        max_history: int = 10,
        compress_min_messages: int = 4,
        system_prompt: str = "你是一个有用的AI助手。",
        verbose: bool = True,
    ):
        self.client = client
        self.model = model
        self.session_id = session_id
        self.verbose = verbose

        self.store = JsonSessionStore(base_dir=store_dir)
        self.context_builder = ContextWindowBuilder(
            max_history=max_history, system_prompt=system_prompt
        )
        self.compressor = RollingSummaryCompressor(
            client=client, model=model, min_messages=compress_min_messages
        )

        self.session = self.store.load(session_id)

    def chat(self, user_input: str) -> str:
        self.session.messages.append({"role": "user", "content": user_input})

        # 自动压缩：消息数超过 max_history + min_messages 时触发
        threshold = self.context_builder.max_history * 2 + self.compressor.min_messages
        if len(self.session.messages) > threshold:
            self._log(f"\n{'='*50}")
            self._log(f"[压缩触发] 消息数 {len(self.session.messages)} > 阈值 {threshold}")
            self._log(f"{'='*50}")

            # 打印压缩前的历史消息
            self._log("\n[压缩前] 历史消息:")
            for i, msg in enumerate(self.session.messages):
                self._log(f"  {i+1}. {msg['role']}: {msg['content']}")

            # 执行压缩
            self.session = self.compressor.compress(self.session)

            # 打印压缩后的摘要
            self._log(f"\n[压缩后] 生成的摘要:")
            self._log(f"  {self.session.compress_content}")

            # 打印压缩后保留的消息
            self._log(f"\n[压缩后] 保留的近期消息 ({len(self.session.messages)} 条):")
            for i, msg in enumerate(self.session.messages):
                self._log(f"  {i+1}. {msg['role']}: {msg['content']}")
            self._log(f"{'='*50}\n")

        # 裁剪上下文窗口
        messages = self.context_builder.build(self.session)

        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        reply = response.choices[0].message.content

        self.session.messages.append({"role": "assistant", "content": reply})
        self.store.save(self.session_id, self.session)

        return reply

    def get_history(self) -> list[dict[str, str]]:
        return self.session.messages

    def get_summary(self) -> str:
        return self.session.compress_content

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
