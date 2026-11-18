"""System prompt plus trimmed chat history. See spec §4.4."""
from __future__ import annotations


class Conversation:
    def __init__(self, system_prompt: str, history_turns: int) -> None:
        self.system_prompt = system_prompt
        self.history_turns = history_turns
        self._turns: list[tuple[str, str]] = []

    @property
    def turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    def messages(self, user_text: str) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        recent = self._turns[-self.history_turns:] if self.history_turns > 0 else []
        for user, assistant in recent:
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": assistant})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def add_exchange(self, user: str, assistant: str) -> None:
        self._turns.append((user, assistant))

    def clear(self) -> None:
        self._turns.clear()
