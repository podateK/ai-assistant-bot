from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import discord
from discord.ext import commands
from loguru import logger

from config import load_settings
from database import Database
from utils.ai_client import AIClient
from utils.memory import ConversationMemory
from utils.prompt_engine import PromptEngine
from utils.rate_limiter import RateLimiter


class AIBot(commands.Bot):
    def __init__(self) -> None:
        self.settings = load_settings()
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True

        super().__init__(
            command_prefix=self.settings.default_prefix,
            intents=intents,
            application_id=self.settings.discord_application_id,
            help_command=None,
        )

        self.db: Database | None = None
        self.ai_client: AIClient | None = None
        self.memory: ConversationMemory | None = None
        self.prompt_engine: PromptEngine | None = None
        self.rate_limiter: RateLimiter | None = None

    async def setup_hook(self) -> None:
        self.db = Database(self.settings.database_path)
        await self.db.connect()

        self.ai_client = AIClient(self.settings)
        await self.ai_client.initialize()

        self.memory = ConversationMemory(self.db, self.settings)
        self.prompt_engine = PromptEngine()
        self.rate_limiter = RateLimiter(self.settings)

        cog_modules = [
            "cogs.ai_chat",
            "cogs.image_gen",
            "cogs.code_helper",
            "cogs.summarizer",
            "cogs.translator",
            "cogs.fun_ai",
        ]
        for module in cog_modules:
            try:
                await self.load_extension(module)
                logger.info("Loaded extension: {}", module)
            except Exception as e:
                logger.error("Failed to load {}: {}", module, e)

        await self.tree.sync()
        logger.info("Synced application commands")

    async def on_ready(self) -> None:
        logger.info("Logged in as {} (ID: {})", self.user, self.user.id if self.user else "Unknown")
        logger.info("Connected to {} guilds", len(self.guilds))

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.reply(f"Missing argument: `{error.param.name}`", mention_author=False)
        elif isinstance(error, commands.CommandOnCooldown):
            await ctx.reply(f"Cooldown. Try again in {error.retry_after:.1f}s.", mention_author=False)
        elif isinstance(error, commands.MissingPermissions):
            await ctx.reply("You don't have permission to use this command.", mention_author=False)
        else:
            logger.error("Command error: {}", error)
            await ctx.reply("An error occurred while processing the command.", mention_author=False)

    async def close(self) -> None:
        if self.ai_client:
            await self.ai_client.close()
        if self.db:
            await self.db.close()
        await super().close()


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        "data/bot.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


async def main() -> None:
    setup_logging()
    logger.info("Starting AI Assistant Bot...")

    bot = AIBot()
    bot.settings.data_dir.mkdir(parents=True, exist_ok=True)

    try:
        await bot.start(bot.settings.discord_token)
    except discord.LoginFailure:
        logger.error("Invalid Discord token provided")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
