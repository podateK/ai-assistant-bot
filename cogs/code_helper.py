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


class CodeHelper(commands.Cog):
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

    def _extract_code(self, text: str) -> tuple[str, str]:
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                first_line = parts[1].split("\n", 1)
                lang = first_line[0].strip() if len(first_line) > 1 else ""
                code = first_line[1] if len(first_line) > 1 else parts[1]
                return code.strip(), lang
        return text.strip(), ""

    @commands.hybrid_command(name="analyze", aliases=["review"])
    @app_commands.describe(
        code="Code to analyze (or paste in code block)",
        language="Programming language",
        task="Specific analysis task",
    )
    async def analyze_code(
        self,
        ctx: commands.Context,
        *,
        code: str,
        language: str | None = None,
        task: str | None = None,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "code"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "code")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        extracted_code, detected_lang = self._extract_code(code)
        lang = language or detected_lang

        async with ctx.typing():
            system, user_prompt = self.prompt_engine.build_code_analysis_prompt(extracted_code, lang, task)
            response = await self.ai.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
            )
            self.rate_limiter.record(ctx.author.id, "code")

            if self.settings.token_tracking_enabled:
                await self.db.track_tokens(ctx.author.id, response.model, response.prompt_tokens, response.completion_tokens, "code_analysis")

        if len(response.content) > 2000:
            file = discord.File(
                fp=__import__("io").BytesIO(response.content.encode()),
                filename="code_analysis.md",
            )
            await ctx.reply("Analysis too long for a message. See attached file:", file=file, mention_author=False)
        else:
            await ctx.reply(response.content, mention_author=False)

    @commands.hybrid_command(name="generate_code", aliases=["codegen"])
    @app_commands.describe(
        task="What the code should do",
        language="Target programming language",
        framework="Framework or library to use",
    )
    async def generate_code(
        self,
        ctx: commands.Context,
        *,
        task: str,
        language: str | None = None,
        framework: str | None = None,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "code"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "code")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        async with ctx.typing():
            system, user_prompt = self.prompt_engine.build_code_generation_prompt(task, language, framework)
            response = await self.ai.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
            )
            self.rate_limiter.record(ctx.author.id, "code")

            if self.settings.token_tracking_enabled:
                await self.db.track_tokens(ctx.author.id, response.model, response.prompt_tokens, response.completion_tokens, "code_generation")

        if len(response.content) > 2000:
            file = discord.File(
                fp=__import__("io").BytesIO(response.content.encode()),
                filename="generated_code.md",
            )
            await ctx.reply("Generated code is long. See attached file:", file=file, mention_author=False)
        else:
            await ctx.reply(response.content, mention_author=False)

    @commands.hybrid_command(name="explain")
    @app_commands.describe(
        code="Code to explain",
        language="Programming language",
    )
    async def explain_code(
        self,
        ctx: commands.Context,
        *,
        code: str,
        language: str | None = None,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "code"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "code")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        extracted_code, detected_lang = self._extract_code(code)
        lang = language or detected_lang

        async with ctx.typing():
            system, user_prompt = self.prompt_engine.render_prompt(
                "code_explain", code=extracted_code, language=lang
            )
            response = await self.ai.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
            )
            self.rate_limiter.record(ctx.author.id, "code")

        if len(response.content) > 2000:
            file = discord.File(
                fp=__import__("io").BytesIO(response.content.encode()),
                filename="code_explanation.md",
            )
            await ctx.reply("Explanation is long. See attached file:", file=file, mention_author=False)
        else:
            await ctx.reply(response.content, mention_author=False)

    @commands.hybrid_command(name="debug")
    @app_commands.describe(
        code="Code with the error",
        error="The error message or traceback",
        language="Programming language",
    )
    async def debug_code(
        self,
        ctx: commands.Context,
        *,
        code: str,
        error: str | None = None,
        language: str | None = None,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "code"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "code")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        extracted_code, detected_lang = self._extract_code(code)
        lang = language or detected_lang

        async with ctx.typing():
            system, user_prompt = self.prompt_engine.render_prompt(
                "debug", code=extracted_code, error=error or "", language=lang
            )
            response = await self.ai.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
            )
            self.rate_limiter.record(ctx.author.id, "code")

        if len(response.content) > 2000:
            file = discord.File(
                fp=__import__("io").BytesIO(response.content.encode()),
                filename="debug_result.md",
            )
            await ctx.reply("Debug analysis is long. See attached file:", file=file, mention_author=False)
        else:
            await ctx.reply(response.content, mention_author=False)

    @commands.hybrid_command(name="code_review")
    @app_commands.describe(code="Code to review")
    async def code_review(self, ctx: commands.Context, *, code: str) -> None:
        if not self.rate_limiter.check(ctx.author.id, "code"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "code")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        extracted_code, detected_lang = self._extract_code(code)

        async with ctx.typing():
            system, user_prompt = self.prompt_engine.build_code_analysis_prompt(
                extracted_code,
                detected_lang,
                task="Perform a thorough code review. Check for bugs, security issues, performance problems, code style, and provide improvement suggestions.",
            )
            response = await self.ai.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
            )
            self.rate_limiter.record(ctx.author.id, "code")

        if len(response.content) > 2000:
            file = discord.File(
                fp=__import__("io").BytesIO(response.content.encode()),
                filename="code_review.md",
            )
            await ctx.reply("Review is long. See attached file:", file=file, mention_author=False)
        else:
            await ctx.reply(response.content, mention_author=False)

    @app_commands.command(name="code_languages", description="Supported programming languages for code analysis")
    async def supported_languages(self, interaction: discord.Interaction) -> None:
        languages = [
            "Python", "JavaScript", "TypeScript", "Java", "C#", "C++", "C",
            "Go", "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala",
            "SQL", "HTML", "CSS", "Shell/Bash", "PowerShell", "R",
            "MATLAB", "Lua", "Dart", "Elixir", "Haskell", "Clojure",
            "Assembly", "Fortran", "COBOL", "Perl",
        ]
        embed = discord.Embed(title="Supported Languages", color=discord.Color.orange())
        embed.description = "\n".join(f"- {lang}" for lang in languages)
        embed.set_footer(text="Language auto-detected from code blocks, or specify with language: parameter")
        await interaction.response.send_message(embed=embed, ephemeral=True)
