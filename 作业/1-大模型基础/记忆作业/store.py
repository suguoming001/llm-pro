import json
from dataclasses import asdict
from pathlib import Path

from models import Session


class JsonSessionStore:
    def __init__(self, base_dir: str = "./sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.json"

    def save(self, session_id: str, session: Session) -> None:
        session.touch()
        path = self._get_path(session_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(session), f, ensure_ascii=False, indent=2)

    def load(self, session_id: str) -> Session:
        path = self._get_path(session_id)
        if not path.exists():
            return Session()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return Session(**data)
