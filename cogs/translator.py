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


SUPPORTED_LANGUAGES = {
    "english": "English",
    "spanish": "Spanish",
    "french": "French",
    "german": "German",
    "italian": "Italian",
    "portuguese": "Portuguese",
    "russian": "Russian",
    "japanese": "Japanese",
    "chinese": "Chinese (Simplified)",
    "korean": "Korean",
    "arabic": "Arabic",
    "hindi": "Hindi",
    "turkish": "Turkish",
    "dutch": "Dutch",
    "polish": "Polish",
    "swedish": "Swedish",
    "danish": "Danish",
    "finnish": "Finnish",
    "norwegian": "Norwegian",
    "czech": "Czech",
    "romanian": "Romanian",
    "hungarian": "Hungarian",
    "thai": "Thai",
    "vietnamese": "Vietnamese",
    "indonesian": "Indonesian",
    "malay": "Malay",
    "filipino": "Filipino",
    "ukrainian": "Ukrainian",
    "greek": "Greek",
    "hebrew": "Hebrew",
    "latin": "Latin",
    "esperanto": "Esperanto",
    "pirate": "Pirate English",
    "shakespeare": "Shakespearean English",
    "emoji": "Emoji",
    "binary": "Binary",
    "base64": "Base64",
}


class Translator(commands.Cog):
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

    @commands.hybrid_command(name="translate", aliases=["tr"])
    @app_commands.describe(
        text="Text to translate",
        target_language="Target language (e.g., spanish, french, japanese)",
        context="Additional context for better translation",
    )
    async def translate(
        self,
        ctx: commands.Context,
        target_language: str,
        *,
        text: str,
        context: str | None = None,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "translate"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "translate")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        lang_name = SUPPORTED_LANGUAGES.get(target_language.lower(), target_language)

        async with ctx.typing():
            system, user_prompt = self.prompt_engine.build_translation_prompt(text, lang_name, context)
            response = await self.ai.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
            )
            self.rate_limiter.record(ctx.author.id, "translate")

            if self.settings.token_tracking_enabled:
                await self.db.track_tokens(
                    ctx.author.id, response.model, response.prompt_tokens, response.completion_tokens, "translate"
                )

        embed = discord.Embed(title="Translation", color=discord.Color.blue())
        embed.add_field(name=f"To {lang_name}", value=response.content[:2000], inline=False)
        embed.set_footer(text="Powered by AI")
        await ctx.reply(embed=embed, mention_author=False)

    @commands.hybrid_command(name="detect_language")
    @app_commands.describe(text="Text to identify the language of")
    async def detect_language(self, ctx: commands.Context, *, text: str) -> None:
        if not self.rate_limiter.check(ctx.author.id, "translate"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "translate")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        async with ctx.typing():
            response = await self.ai.generate(
                messages=[{"role": "user", "content": f"Identify the language of this text. Reply with just the language name:\n\n{text[:500]}"}],
                system="You are a language detection expert. Reply with only the language name.",
                max_tokens=50,
                temperature=0.1,
            )
            self.rate_limiter.record(ctx.author.id, "translate")

        await ctx.reply(f"Detected language: **{response.content.strip()}**", mention_author=False)

    @commands.hybrid_command(name="define")
    @app_commands.describe(word="Word or phrase to define")
    async def define_word(self, ctx: commands.Context, *, word: str) -> None:
        if not self.rate_limiter.check(ctx.author.id, "translate"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "translate")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        async with ctx.typing():
            response = await self.ai.generate(
                messages=[{
                    "role": "user",
                    "content": f"Define '{word}'. Include: definition, pronunciation guide, etymology (brief), example usage, and synonyms.",
                }],
                system="You are a dictionary expert. Provide clear, concise definitions.",
                temperature=0.3,
            )
            self.rate_limiter.record(ctx.author.id, "translate")

        embed = discord.Embed(title=f"Definition: {word}", description=response.content[:2000], color=discord.Color.gold())
        await ctx.reply(embed=embed, mention_author=False)

    @commands.hybrid_command(name="synonyms")
    @app_commands.describe(word="Word to find synonyms for")
    async def find_synonyms(self, ctx: commands.Context, *, word: str) -> None:
        if not self.rate_limiter.check(ctx.author.id, "translate"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "translate")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        async with ctx.typing():
            response = await self.ai.generate(
                messages=[{"role": "user", "content": f"List 10 synonyms for '{word}' grouped by similarity. Include a brief note on nuance differences."}],
                system="You are a thesaurus expert.",
                temperature=0.3,
            )
            self.rate_limiter.record(ctx.author.id, "translate")

        embed = discord.Embed(title=f"Synonyms for: {word}", description=response.content[:2000], color=discord.Color.gold())
        await ctx.reply(embed=embed, mention_author=False)

    @app_commands.command(name="languages", description="List all supported translation languages")
    async def list_languages(self, interaction: discord.Interaction) -> None:
        embed = discord.Embed(title="Supported Languages", color=discord.Color.blue())
        normal = [f"• {v}" for k, v in SUPPORTED_LANGUAGES.items() if k not in ("pirate", "shakespeare", "emoji", "binary", "base64")]
        fun = [f"• {v}" for k, v in SUPPORTED_LANGUAGES.items() if k in ("pirate", "shakespeare", "emoji", "binary", "base64")]
        embed.add_field(name="Languages", value="\n".join(normal[:25]) or "None", inline=True)
        embed.add_field(name="Fun Translations", value="\n".join(fun) or "None", inline=True)
        embed.set_footer(text="Use /translate text: ... target_language: <language>")
        await interaction.response.send_message(embed=embed, ephemeral=True)
