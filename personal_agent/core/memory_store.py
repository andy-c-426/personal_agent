import json
import time
from pathlib import Path


class MemoryStore:
    """Persistent user memory: preferences, facts, things to remember."""

    def __init__(self, path: Path):
        self._path = path
        self._items: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._items = json.loads(self._path.read_text())
            except Exception:
                self._items = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._items, ensure_ascii=False, indent=2))

    def add(self, text: str) -> dict:
        item = {
            "id": str(int(time.time() * 1_000_000)),
            "text": text.strip(),
            "created_at": time.time(),
        }
        self._items.append(item)
        self._save()
        return item

    def list_all(self) -> list[dict]:
        return list(self._items)

    def remove(self, item_id: str) -> bool:
        before = len(self._items)
        self._items = [m for m in self._items if m["id"] != item_id]
        if len(self._items) < before:
            self._save()
            return True
        return False

    def count(self) -> int:
        return len(self._items)

    def format_for_prompt(self) -> str:
        if not self._items:
            return ""
        lines = ["## User Memory", ""]
        for m in self._items:
            lines.append(f"- {m['text']}")
        return "\n".join(lines)
