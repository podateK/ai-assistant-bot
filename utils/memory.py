from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from config import Settings
from database import Database


@dataclass
class ConversationMessage:
    role: str
    content: str
    tokens_used: int = 0


@dataclass
class MemoryContext:
    summary: str | None = None
    recent_messages: list[ConversationMessage] = field(default_factory=list)
    total_tokens_estimated: int = 0


class ConversationMemory:
    def __init__(self, db: Database, settings: Settings) -> None:
        self._db = db
        self._settings = settings
        self._cache: dict[tuple[int, int], list[ConversationMessage]] = {}

    def _cache_key(self, user_id: int, channel_id: int) -> tuple[int, int]:
        return (user_id, channel_id)

    async def add_message(
        self,
        user_id: int,
        channel_id: int,
        guild_id: int,
        role: str,
        content: str,
        model: str,
        tokens_used: int = 0,
    ) -> None:
        await self._db.add_message(user_id, channel_id, guild_id, role, content, model, tokens_used)

        key = self._cache_key(user_id, channel_id)
        if key not in self._cache:
            self._cache[key] = []
        self._cache[key].append(ConversationMessage(role=role, content=content, tokens_used=tokens_used))

    async def get_context(
        self,
        user_id: int,
        channel_id: int,
        max_tokens: int = 8000,
    ) -> MemoryContext:
        key = self._cache_key(user_id, channel_id)
        cached = self._cache.get(key, [])

        if len(cached) >= self._settings.conversation_summary_threshold:
            await self._summarize_and_trim(user_id, channel_id, guild_id=0)
            cached = self._cache.get(key, [])

        messages = await self._db.get_messages(user_id, channel_id, self._settings.max_conversation_history)

        summary = await self._db.get_latest_summary(user_id, channel_id)

        context = MemoryContext(summary=summary)
        total_tokens = 0

        if summary:
            total_tokens += len(summary) // 4

        recent = []
        for msg in messages:
            msg_tokens = len(msg["content"]) // 4
            if total_tokens + msg_tokens > max_tokens:
                break
            total_tokens += msg_tokens
            recent.append(
                ConversationMessage(
                    role=msg["role"],
                    content=msg["content"],
                    tokens_used=msg.get("tokens_used", 0),
                )
            )

        context.recent_messages = recent
        context.total_tokens_estimated = total_tokens
        return context

    async def _summarize_and_trim(
        self,
        user_id: int,
        channel_id: int,
        guild_id: int = 0,
    ) -> None:
        messages = await self._db.get_messages(user_id, channel_id, self._settings.max_conversation_history)

        if len(messages) < self._settings.conversation_summary_threshold:
            return

        conversation_text = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages[:self._settings.conversation_summary_threshold]
        )

        summary_prompt = [
            {
                "role": "user",
                "content": f"Summarize the following conversation concisely, keeping key context and decisions:\n\n{conversation_text}",
            }
        ]

        try:
            from utils.ai_client import AIClient

            ai_client = AIClient(self._settings)
            await ai_client.initialize()

            response = await ai_client.generate(
                messages=summary_prompt,
                max_tokens=500,
                temperature=0.3,
                system="You are a concise summarizer. Preserve key context, decisions, and user preferences.",
            )
            await ai_client.close()

            await self._db.save_summary(
                user_id=user_id,
                channel_id=channel_id,
                guild_id=guild_id,
                summary=response.content,
                message_count=self._settings.conversation_summary_threshold,
                model=response.model,
            )

            key = self._cache_key(user_id, channel_id)
            remaining = messages[self._settings.conversation_summary_threshold:]
            self._cache[key] = [
                ConversationMessage(role=m["role"], content=m["content"], tokens_used=m.get("tokens_used", 0))
                for m in remaining
            ]

            logger.info(
                "Summarized {} messages for user {} in channel {}",
                self._settings.conversation_summary_threshold,
                user_id,
                channel_id,
            )
        except Exception as e:
            logger.error("Failed to summarize conversation: {}", e)

    async def clear(self, user_id: int, channel_id: int) -> int:
        count = await self._db.clear_messages(user_id, channel_id)
        key = self._cache_key(user_id, channel_id)
        self._cache.pop(key, None)
        return count

    async def export_to_dict(self, user_id: int, channel_id: int) -> dict:
        messages = await self._db.get_messages(user_id, channel_id, limit=500)
        summary = await self._db.get_latest_summary(user_id, channel_id)
        return {
            "user_id": user_id,
            "channel_id": channel_id,
            "summary": summary,
            "messages": messages,
            "message_count": len(messages),
        }

    async def export_to_json(self, user_id: int, channel_id: int) -> str:
        import json

        data = await self.export_to_dict(user_id, channel_id)
        return json.dumps(data, indent=2, default=str)

    async def build_messages_for_ai(
        self,
        user_id: int,
        channel_id: int,
        new_message: str,
        system_prompt: str | None = None,
        max_tokens: int = 8000,
    ) -> list[dict[str, str]]:
        context = await self.get_context(user_id, channel_id, max_tokens)

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context.summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{context.summary}",
            })

        for msg in context.recent_messages:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": new_message})
        return messages

    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
