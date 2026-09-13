from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiosqlite

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    guild_id INTEGER DEFAULT 0,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    guild_id INTEGER DEFAULT 0,
    summary TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    model TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    feature TEXT DEFAULT 'chat',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_limits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    timestamp REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS image_generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    revised_prompt TEXT DEFAULT '',
    url TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS exported_conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    format TEXT NOT NULL,
    file_path TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_user_channel ON conversations(user_id, channel_id);
CREATE INDEX IF NOT EXISTS idx_conv_guild ON conversations(guild_id);
CREATE INDEX IF NOT EXISTS idx_memory_user_channel ON conversation_memory(user_id, channel_id);
CREATE INDEX IF NOT EXISTS idx_token_user ON token_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_rate_user_endpoint ON rate_limits(user_id, endpoint);
CREATE INDEX IF NOT EXISTS idx_image_user ON image_generations(user_id);
"""


@dataclass
class Database:
    db_path: str
    _db: aiosqlite.Connection | None = field(default=None, repr=False)

    async def connect(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(CREATE_TABLES_SQL)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if not self._db:
            raise RuntimeError("Database not connected")
        return self._db

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
        await self.db.execute(
            "INSERT INTO conversations (user_id, channel_id, guild_id, role, content, model, tokens_used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, channel_id, guild_id, role, content, model, tokens_used, time.time()),
        )
        await self.db.commit()

    async def get_messages(
        self,
        user_id: int,
        channel_id: int,
        limit: int = 50,
    ) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT role, content, tokens_used, created_at FROM conversations WHERE user_id = ? AND channel_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, channel_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in reversed(rows)]

    async def save_summary(
        self,
        user_id: int,
        channel_id: int,
        guild_id: int,
        summary: str,
        message_count: int,
        model: str,
    ) -> None:
        await self.db.execute(
            "INSERT INTO conversation_memory (user_id, channel_id, guild_id, summary, message_count, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, channel_id, guild_id, summary, message_count, model, time.time()),
        )
        await self.db.commit()

    async def get_latest_summary(self, user_id: int, channel_id: int) -> str | None:
        cursor = await self.db.execute(
            "SELECT summary FROM conversation_memory WHERE user_id = ? AND channel_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id, channel_id),
        )
        row = await cursor.fetchone()
        return row["summary"] if row else None

    async def clear_messages(self, user_id: int, channel_id: int) -> int:
        cursor = await self.db.execute(
            "DELETE FROM conversations WHERE user_id = ? AND channel_id = ?",
            (user_id, channel_id),
        )
        await self.db.commit()
        return cursor.rowcount

    async def track_tokens(
        self,
        user_id: int,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        feature: str = "chat",
    ) -> None:
        total = prompt_tokens + completion_tokens
        await self.db.execute(
            "INSERT INTO token_usage (user_id, model, prompt_tokens, completion_tokens, total_tokens, feature, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, model, prompt_tokens, completion_tokens, total, feature, time.time()),
        )
        await self.db.commit()

    async def get_user_token_usage(self, user_id: int, days: int = 30) -> dict:
        cutoff = time.time() - (days * 86400)
        cursor = await self.db.execute(
            "SELECT model, SUM(prompt_tokens) as prompt, SUM(completion_tokens) as completion, SUM(total_tokens) as total, COUNT(*) as requests FROM token_usage WHERE user_id = ? AND created_at > ? GROUP BY model",
            (user_id, cutoff),
        )
        rows = await cursor.fetchall()
        return {
            row["model"]: {
                "prompt_tokens": row["prompt"],
                "completion_tokens": row["completion"],
                "total_tokens": row["total"],
                "requests": row["requests"],
            }
            for row in rows
        }

    async def check_rate_limit(self, user_id: int, endpoint: str, window: float = 60.0) -> int:
        cutoff = time.time() - window
        cursor = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM rate_limits WHERE user_id = ? AND endpoint = ? AND timestamp > ?",
            (user_id, endpoint, cutoff),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def record_rate_limit(self, user_id: int, endpoint: str) -> None:
        await self.db.execute(
            "INSERT INTO rate_limits (user_id, endpoint, timestamp) VALUES (?, ?, ?)",
            (user_id, endpoint, time.time()),
        )
        await self.db.commit()

    async def cleanup_old_rate_limits(self, max_age: float = 300.0) -> None:
        cutoff = time.time() - max_age
        await self.db.execute(
            "DELETE FROM rate_limits WHERE timestamp < ?",
            (cutoff,),
        )
        await self.db.commit()

    async def save_image_generation(
        self,
        user_id: int,
        prompt: str,
        revised_prompt: str,
        url: str,
        model: str,
    ) -> None:
        await self.db.execute(
            "INSERT INTO image_generations (user_id, prompt, revised_prompt, url, model, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, prompt, revised_prompt, url, model, time.time()),
        )
        await self.db.commit()

    async def get_image_history(self, user_id: int, limit: int = 10) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT prompt, url, model, created_at FROM image_generations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def save_export(
        self,
        user_id: int,
        channel_id: int,
        format_type: str,
        file_path: str,
        message_count: int,
    ) -> None:
        await self.db.execute(
            "INSERT INTO exported_conversations (user_id, channel_id, format, file_path, message_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, channel_id, format_type, file_path, message_count, time.time()),
        )
        await self.db.commit()

    async def get_guild_message_count(self, guild_id: int) -> int:
        cursor = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM conversations WHERE guild_id = ?",
            (guild_id,),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def get_global_stats(self) -> dict:
        stats = {}
        for table in ["conversations", "conversation_memory", "token_usage", "image_generations"]:
            cursor = await self.db.execute(f"SELECT COUNT(*) as cnt FROM {table}")
            row = await cursor.fetchone()
            stats[f"total_{table}"] = row["cnt"]
        cursor = await self.db.execute("SELECT SUM(total_tokens) as total FROM token_usage")
        row = await cursor.fetchone()
        stats["total_tokens_all_time"] = row["total"] or 0
        return stats
