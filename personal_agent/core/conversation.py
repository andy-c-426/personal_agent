import json
import tiktoken
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # system, user, assistant, tool
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    @property
    def token_count(self) -> int:
        enc = tiktoken.get_encoding("cl100k_base")
        text = self.content or ""
        if self.tool_calls:
            text += json.dumps(self.tool_calls)
        return len(enc.encode(text))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    summary: str = ""

    def add_message(
        self,
        role: str,
        content: str,
        name: str | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ) -> Message:
        msg = Message(role=role, content=content, name=name, tool_calls=tool_calls, tool_call_id=tool_call_id)
        self.messages.append(msg)
        return msg

    def to_dicts(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.messages]

    @property
    def total_tokens(self) -> int:
        return sum(m.token_count for m in self.messages)

    def trim_to_fit(self, max_messages: int) -> int:
        dropped = 0
        while len(self.messages) > max_messages:
            self.messages.pop(0)
            dropped += 1
        return dropped

    def to_json(self) -> str:
        data = {
            "messages": [m.to_dict() for m in self.messages],
            "summary": self.summary,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "Conversation":
        obj = json.loads(data)
        conv = cls(summary=obj.get("summary", ""))
        for m in obj.get("messages", []):
            conv.add_message(
                role=m["role"],
                content=m.get("content", ""),
                name=m.get("name"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
        return conv
