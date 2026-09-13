from __future__ import annotations

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from config import Settings
from database import Database
from utils.ai_client import AIClient, ImageResponse
from utils.memory import ConversationMemory
from utils.prompt_engine import PromptEngine
from utils.rate_limiter import RateLimiter


class ImageGeneration(commands.Cog):
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

    @commands.hybrid_command(name="imagine", aliases=["dalle", "generate"])
    @app_commands.describe(
        prompt="Description of the image to generate",
        style="Artistic style (e.g., photorealistic, anime, oil painting)",
        mood="Mood or atmosphere (e.g., dark, bright, dreamy)",
    )
    async def generate_image(
        self,
        ctx: commands.Context,
        *,
        prompt: str,
        style: str | None = None,
        mood: str | None = None,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "image"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "image")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        await ctx.reply("Generating image...", mention_author=False)

        try:
            system, user_prompt = self.prompt_engine.build_image_prompt(prompt, style, mood)
            enhancement_response = await self.ai.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
                max_tokens=300,
                temperature=0.9,
            )
            final_prompt = enhancement_response.content.strip()
        except Exception:
            final_prompt = prompt

        try:
            image_response = await self.ai.generate_image(prompt=final_prompt)
            await self._send_image_result(ctx, image_response, final_prompt, prompt)
            self.rate_limiter.record(ctx.author.id, "image")

            if self.settings.token_tracking_enabled:
                await self.db.track_tokens(
                    ctx.author.id, image_response.model, 0, 0, "image_generation"
                )
                await self.db.save_image_generation(
                    ctx.author.id, prompt, image_response.revised_prompt, image_response.url, image_response.model
                )
        except Exception as e:
            await ctx.reply(f"Image generation failed: {e}", mention_author=False)

    async def _send_image_result(
        self,
        ctx: commands.Context,
        image_response: ImageResponse,
        enhanced_prompt: str,
        original_prompt: str,
    ) -> None:
        embed = discord.Embed(
            title="AI Generated Image",
            color=discord.Color.purple(),
        )
        embed.set_image(url=image_response.url)
        embed.add_field(name="Original", value=original_prompt[:1000], inline=False)
        if enhanced_prompt != original_prompt:
            embed.add_field(name="Enhanced Prompt", value=enhanced_prompt[:1000], inline=False)
        embed.set_footer(text=f"Model: {image_response.model}")

        await ctx.reply(embed=embed, mention_author=False)

        async with aiohttp.ClientSession() as session:
            async with session.get(image_response.url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    file = discord.File(fp=__import__("io").BytesIO(data), filename="ai_image.png")
                    await ctx.reply(file=file, mention_author=False)

    @commands.hybrid_command(name="image_history")
    async def image_history(self, ctx: commands.Context, limit: int = 5) -> None:
        history = await self.db.get_image_history(ctx.author.id, limit)
        if not history:
            await ctx.reply("No image generation history found.", mention_author=False)
            return

        embed = discord.Embed(title="Image Generation History", color=discord.Color.purple())
        for i, entry in enumerate(history, 1):
            embed.add_field(
                name=f"#{i} - {entry['model']}",
                value=f"**Prompt:** {entry['prompt'][:100]}...\n[View Image]({entry['url']})",
                inline=False,
            )
        await ctx.reply(embed=embed, mention_author=False)

    @app_commands.command(name="image_styles", description="List available image styles")
    async def image_styles(self, interaction: discord.Interaction) -> None:
        styles = [
            "Photorealistic", "Oil Painting", "Watercolor", "Anime", "Pixel Art",
            "3D Render", "Pencil Sketch", "Digital Art", "Abstract", "Cyberpunk",
            "Steampunk", "Fantasy", "Sci-Fi", "Horror", "Pop Art",
        ]
        embed = discord.Embed(title="Image Styles", color=discord.Color.purple())
        embed.description = "\n".join(f"- {s}" for s in styles)
        embed.set_footer(text="Use with /imagine prompt: ... style: <style>")
        await interaction.response.send_message(embed=embed, ephemeral=True)
