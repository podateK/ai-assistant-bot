from __future__ import annotations

import re

import aiohttp
import discord
from bs4 import BeautifulSoup
from discord import app_commands
from discord.ext import commands

from config import Settings
from database import Database
from utils.ai_client import AIClient
from utils.memory import ConversationMemory
from utils.prompt_engine import PromptEngine
from utils.rate_limiter import RateLimiter


class Summarizer(commands.Cog):
    def __init__(
        self,
        bot: commands.Bot,
        db: Database,
        ai_client: AIClient,
        memory: ConversationMemory,
        prompt_engine: PromptEngine,
        rate_limiter: RateLimiter,
        settings: Settings,
    ) -> None:
        self.bot = bot
        self.db = db
        self.ai = ai_client
        self.memory = memory
        self.prompt_engine = prompt_engine
        self.rate_limiter = rate_limiter
        self.settings = settings

    async def _fetch_url_content(self, url: str) -> str | None:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        return None
                    html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return "\n".join(chunk for chunk in chunks if chunk)[:15000]
        except Exception:
            return None

    @commands.hybrid_command(name="summarize", aliases=["summary", "tldr"])
    @app_commands.describe(
        text="Text to summarize (or URL to fetch)",
        length="Summary length: short, medium, or long",
    )
    async def summarize(
        self,
        ctx: commands.Context,
        *,
        text: str,
        length: str | None = None,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "summarize"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "summarize")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        url_match = re.search(r"https?://[^\s]+", text)
        url = None
        content = text

        if url_match:
            url = url_match.group(0)
            await ctx.reply(f"Fetching content from {url}...", mention_author=False)
            fetched = await self._fetch_url_content(url)
            if not fetched:
                await ctx.reply("Failed to fetch URL content.", mention_author=False)
                return
            content = fetched

        async with ctx.typing():
            system, user_prompt = self.prompt_engine.build_summarization_prompt(content, length, url)
            response = await self.ai.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
            )
            self.rate_limiter.record(ctx.author.id, "summarize")

            if self.settings.token_tracking_enabled:
                await self.db.track_tokens(
                    ctx.author.id, response.model, response.prompt_tokens, response.completion_tokens, "summarize"
                )

        embed = discord.Embed(title="Summary", color=discord.Color.teal())
        if url:
            embed.url = url
            embed.add_field(name="Source", value=url[:200], inline=False)

        if len(response.content) > 2000:
            file = discord.File(
                fp=__import__("io").BytesIO(response.content.encode()),
                filename="summary.md",
            )
            await ctx.reply(embed=embed, file=file, mention_author=False)
        else:
            embed.description = response.content
            await ctx.reply(embed=embed, mention_author=False)

    @commands.hybrid_command(name="summarize_file")
    @app_commands.describe(
        length="Summary length: short, medium, or long",
    )
    async def summarize_attachment(
        self,
        ctx: commands.Context,
        length: str | None = None,
    ) -> None:
        if not ctx.message.attachments:
            await ctx.reply("Please attach a text file to summarize.", mention_author=False)
            return

        attachment = ctx.message.attachments[0]
        if attachment.size > 1_000_000:
            await ctx.reply("File too large (max 1MB).", mention_author=False)
            return

        content = await attachment.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            await ctx.reply("Could not decode file. Please provide a text file.", mention_author=False)
            return

        if not self.rate_limiter.check(ctx.author.id, "summarize"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "summarize")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        async with ctx.typing():
            system, user_prompt = self.prompt_engine.build_summarization_prompt(text[:15000], length)
            response = await self.ai.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
            )
            self.rate_limiter.record(ctx.author.id, "summarize")

        embed = discord.Embed(
            title=f"Summary of {attachment.filename}",
            description=response.content[:2000] if len(response.content) <= 2000 else "See attached file",
            color=discord.Color.teal(),
        )

        if len(response.content) > 2000:
            file = discord.File(
                fp=__import__("io").BytesIO(response.content.encode()),
                filename="summary.md",
            )
            await ctx.reply(embed=embed, file=file, mention_author=False)
        else:
            await ctx.reply(embed=embed, mention_author=False)

    @app_commands.command(name="quick_summary", description="Get a TL;DR of provided text")
    async def quick_summary(self, interaction: discord.Interaction, text: str) -> None:
        await interaction.response.defer(thinking=True)

        system, user_prompt = self.prompt_engine.build_summarization_prompt(text, "short")
        response = await self.ai.generate(
            messages=[{"role": "user", "content": user_prompt}],
            system=system,
        )

        await interaction.followup.send(f"**TL;DR:** {response.content}")
