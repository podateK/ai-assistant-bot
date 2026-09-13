from __future__ import annotations

import discord
from discord import app_commands
from discord.ext import commands

from config import Settings
from database import Database
from utils.ai_client import AIClient
from utils.memory import ConversationMemory
from utils.prompt_engine import PromptEngine
from utils.rate_limiter import RateLimiter


class AIChat(commands.Cog):
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

    @commands.command(name="chat", aliases=["ask", "ai"])
    async def chat_command(self, ctx: commands.Context, *, message: str) -> None:
        if not ctx.guild:
            return

        if not self.rate_limiter.check(ctx.author.id, "chat"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "chat")
            await ctx.reply(
                f"Rate limit reached. Try again in {info['wait_seconds']}s.",
                mention_author=False,
            )
            return

        async with ctx.typing():
            system = self.prompt_engine.get_template("general").system
            messages = await self.memory.build_messages_for_ai(
                user_id=ctx.author.id,
                channel_id=ctx.channel.id,
                new_message=message,
                system_prompt=system,
            )

            response = await self.ai.generate(messages=messages, provider=self.settings.default_ai_model)

            await self.memory.add_message(
                ctx.author.id, ctx.channel.id, ctx.guild.id, "user", message, response.model, response.prompt_tokens
            )
            await self.memory.add_message(
                ctx.author.id, ctx.channel.id, ctx.guild.id, "assistant", response.content, response.model, response.completion_tokens
            )
            self.rate_limiter.record(ctx.author.id, "chat")

            if self.settings.token_tracking_enabled:
                await self.db.track_tokens(
                    ctx.author.id, response.model, response.prompt_tokens, response.completion_tokens, "chat"
                )

        if len(response.content) > 2000:
            for i in range(0, len(response.content), 2000):
                await ctx.reply(response.content[i : i + 2000], mention_author=False)
        else:
            await ctx.reply(response.content, mention_author=False)

    @commands.hybrid_command(name="clear_history", aliases=["clear"])
    async def clear_history(self, ctx: commands.Context) -> None:
        count = await self.memory.clear(ctx.author.id, ctx.channel.id)
        await ctx.reply(f"Cleared {count} messages from this channel.", mention_author=False)

    @commands.hybrid_command(name="memory")
    async def memory_info(self, ctx: commands.Context) -> None:
        context = await self.memory.get_context(ctx.author.id, ctx.channel.id)
        embed = discord.Embed(title="Conversation Memory", color=discord.Color.blue())
        embed.add_field(name="Summary", value=context.summary or "No summary yet", inline=False)
        embed.add_field(name="Recent Messages", value=str(len(context.recent_messages)), inline=True)
        embed.add_field(name="Est. Tokens", value=str(context.total_tokens_estimated), inline=True)
        await ctx.reply(embed=embed, mention_author=False)

    @commands.hybrid_command(name="export")
    async def export_conversation(self, ctx: commands.Context) -> None:
        json_data = await self.memory.export_to_json(ctx.author.id, ctx.channel.id)
        file = discord.File(
            fp=__import__("io").BytesIO(json_data.encode()),
            filename=f"conversation_{ctx.channel.id}.json",
        )
        await ctx.reply("Here is your conversation export:", file=file, mention_author=False)

    @commands.command(name="stream")
    async def stream_chat(self, ctx: commands.Context, *, message: str) -> None:
        if not ctx.guild:
            return

        if not self.rate_limiter.check(ctx.author.id, "chat"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "chat")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        system = self.prompt_engine.get_template("general").system
        messages = await self.memory.build_messages_for_ai(
            user_id=ctx.author.id,
            channel_id=ctx.channel.id,
            new_message=message,
            system_prompt=system,
        )

        sent_message = await ctx.reply("Generating...", mention_author=False)

        full_response = []
        buffer = ""

        async for chunk in self.ai.stream_generate(messages=messages):
            full_response.append(chunk)
            buffer += chunk
            if len(buffer) >= 100:
                try:
                    await sent_message.edit(content=buffer + "▌")
                except discord.HTTPException:
                    pass
                buffer = ""

        final_content = "".join(full_response)
        if buffer:
            final_content += buffer

        try:
            await sent_message.edit(content=final_content[:2000] if final_content else "No response generated.")
        except discord.HTTPException:
            pass

        await self.memory.add_message(ctx.author.id, ctx.channel.id, ctx.guild.id, "user", message, "stream", 0)
        await self.memory.add_message(ctx.author.id, ctx.channel.id, ctx.guild.id, "assistant", final_content, "stream", 0)
        self.rate_limiter.record(ctx.author.id, "chat")

    @app_commands.command(name="ai_settings", description="View AI settings for this channel")
    async def ai_settings_view(self, interaction: discord.Interaction) -> None:
        embed = discord.Embed(title="AI Settings", color=discord.Color.green())
        embed.add_field(name="Default Model", value=self.settings.default_ai_model, inline=True)
        embed.add_field(name="Rate Limit", value=f"{self.settings.rate_limit_per_minute}/min", inline=True)
        embed.add_field(name="Max History", value=str(self.settings.max_conversation_history), inline=True)
        embed.add_field(
            name="Available Providers",
            value=", ".join(p.value for p in self.settings.available_providers) or "None configured",
            inline=False,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
