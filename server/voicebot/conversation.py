"""System prompt plus trimmed chat history. See spec §4.4 (v3 RAG: spec §4.3)."""
from __future__ import annotations

from voicebot.config import DEFAULT_RAG_TEMPLATE


class Conversation:
    def __init__(self, system_prompt: str, history_turns: int, *,
                 rag_template: str = DEFAULT_RAG_TEMPLATE) -> None:
        self.system_prompt = system_prompt
        self.rag_template = rag_template
        self.history_turns = history_turns
        self._turns: list[tuple[str, str]] = []

    @property
    def turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    def messages(self, user_text: str, context: str = "") -> list[dict[str, str]]:
        """Context goes into this request's system message only; history never stores it.
        The template is filled with str.replace, not str.format, so other braces are kept."""
        system = self.system_prompt
        if context:
            system += "\n\n" + self.rag_template.replace("{context}", context)
        msgs = [{"role": "system", "content": system}]
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
