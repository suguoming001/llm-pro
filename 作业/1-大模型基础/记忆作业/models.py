from dataclasses import dataclass, field
import time


@dataclass
class Session:
    title: str = ""
    create_time: int = field(default_factory=lambda: int(time.time()))
    update_time: int = field(default_factory=lambda: int(time.time()))
    messages: list[dict[str, str]] = field(default_factory=list)
    compress_content: str = ""

    def touch(self) -> None:
        self.update_time = int(time.time())
