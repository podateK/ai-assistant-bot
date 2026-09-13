from __future__ import annotations

import random

import discord
from discord import app_commands
from discord.ext import commands

from config import Settings
from database import Database
from utils.ai_client import AIClient
from utils.memory import ConversationMemory, ConversationMessage
from utils.prompt_engine import PERSONAS, PromptEngine
from utils.rate_limiter import RateLimiter


class FunAI(commands.Cog):
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
        self.active_sessions: dict[int, str] = {}

    @commands.hybrid_command(name="persona", aliases=["character"])
    @app_commands.describe(
        name="Persona name (pirate, robot, wizard, detective, chef, scientist, comedian, philosopher, coach, historian)",
        message="Your message to the persona",
    )
    async def persona_chat(
        self,
        ctx: commands.Context,
        name: str,
        *,
        message: str,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "fun"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "fun")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        persona_key = name.lower().replace(" ", "_")
        if persona_key not in PERSONAS:
            await ctx.reply(
                f"Unknown persona. Available: {', '.join(PERSONAS.keys())}",
                mention_author=False,
            )
            return

        async with ctx.typing():
            system = self.prompt_engine.get_persona_system(persona_key)
            response = await self.ai.generate(
                messages=[{"role": "user", "content": message}],
                system=system,
            )
            self.rate_limiter.record(ctx.author.id, "fun")

            if self.settings.token_tracking_enabled:
                await self.db.track_tokens(
                    ctx.author.id, response.model, response.prompt_tokens, response.completion_tokens, "persona"
                )

        embed = discord.Embed(
            title=f"Speaking with {name.title()}",
            description=response.content,
            color=discord.Color.random(),
        )
        embed.set_footer(text=f"Persona: {persona_key}")
        await ctx.reply(embed=embed, mention_author=False)

    @commands.hybrid_command(name="roleplay")
    @app_commands.describe(
        scenario="The roleplay scenario",
        message="Your character's action or dialogue",
    )
    async def roleplay(
        self,
        ctx: commands.Context,
        scenario: str,
        *,
        message: str,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "fun"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "fun")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        key = f"roleplay_{ctx.author.id}_{ctx.channel.id}"
        if key not in self.active_sessions:
            self.active_sessions[key] = scenario

        system = (
            f"You are a creative roleplay narrator. Scenario: {scenario}. "
            "Write immersive, descriptive responses. Stay in character and advance the story. "
            "Keep responses concise but vivid. Never break character."
        )

        history = []
        if ctx.guild:
            messages = await self.memory.get_context(ctx.author.id, ctx.channel.id)
            for msg in messages.recent_messages[-10:]:
                if msg.role in ("user", "assistant"):
                    history.append({"role": msg.role, "content": msg.content})

        history.append({"role": "user", "content": message})

        async with ctx.typing():
            response = await self.ai.generate(
                messages=history,
                system=system,
                temperature=0.9,
            )
            self.rate_limiter.record(ctx.author.id, "fun")

        embed = discord.Embed(
            title="Roleplay",
            description=response.content,
            color=discord.Color.dark_purple(),
        )
        embed.set_footer(text=f"Scenario: {scenario[:100]}")
        await ctx.reply(embed=embed, mention_author=False)

    @commands.hybrid_command(name="trivia")
    @app_commands.describe(
        category="Trivia category (science, history, geography, pop culture, technology, etc.)",
        difficulty="Difficulty level (easy, medium, hard)",
    )
    async def trivia_game(
        self,
        ctx: commands.Context,
        category: str | None = None,
        difficulty: str | None = None,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "fun"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "fun")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        system, user_prompt = self.prompt_engine.get_trivia_prompt(category, difficulty)

        async with ctx.typing():
            response = await self.ai.generate(
                messages=[{"role": "user", "content": user_prompt}],
                system=system,
            )
            self.rate_limiter.record(ctx.author.id, "fun")

        embed = discord.Embed(
            title=f"Trivia - {category or 'General'} ({difficulty or 'Medium'})",
            description=response.content,
            color=discord.Color.orange(),
        )
        await ctx.reply(embed=embed, mention_author=False)

    @commands.hybrid_command(name="story")
    @app_commands.describe(
        prompt="Story premise or opening line",
        genre="Genre (fantasy, sci-fi, horror, romance, mystery, comedy)",
        length="Story length: short, medium, or long",
    )
    async def generate_story(
        self,
        ctx: commands.Context,
        *,
        prompt: str,
        genre: str | None = None,
        length: str | None = None,
    ) -> None:
        if not self.rate_limiter.check(ctx.author.id, "fun"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "fun")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        length_tokens = {"short": 800, "medium": 2000, "long": 4000}.get(length or "medium", 2000)

        system = (
            f"You are a skilled {genre or 'creative'} fiction writer. "
            "Write an engaging, well-structured story with vivid descriptions and compelling characters. "
            "Maintain a consistent tone and build toward a satisfying conclusion."
        )

        async with ctx.typing():
            response = await self.ai.generate(
                messages=[{"role": "user", "content": f"Write a story: {prompt}"}],
                system=system,
                max_tokens=length_tokens,
                temperature=0.9,
            )
            self.rate_limiter.record(ctx.author.id, "fun")

        if len(response.content) > 2000:
            chunks = [response.content[i : i + 2000] for i in range(0, len(response.content), 2000)]
            for i, chunk in enumerate(chunks):
                prefix = "**Story:**\n" if i == 0 else ""
                await ctx.reply(prefix + chunk, mention_author=False)
        else:
            embed = discord.Embed(
                title=f"{(genre or 'Story').title()}",
                description=response.content,
                color=discord.Color.random(),
            )
            await ctx.reply(embed=embed, mention_author=False)

    @commands.hybrid_command(name="joke")
    async def tell_joke(self, ctx: commands.Context) -> None:
        if not self.rate_limiter.check(ctx.author.id, "fun"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "fun")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        topics = [
            "programming", "animals", "food", "space", "work",
            "school", "technology", "relationships", "sports", "music",
        ]
        topic = random.choice(topics)

        async with ctx.typing():
            response = await self.ai.generate(
                messages=[{"role": "user", "content": f"Tell me a funny {topic} joke. Just the joke, no explanation."}],
                system="You are a stand-up comedian. Tell clean, clever jokes.",
                max_tokens=200,
                temperature=1.0,
            )
            self.rate_limiter.record(ctx.author.id, "fun")

        await ctx.reply(response.content, mention_author=False)

    @commands.hybrid_command(name="roast")
    @app_commands.describe(target="Who to roast (optional, defaults to the command user)")
    async def roast_user(self, ctx: commands.Context, target: discord.Member | None = None) -> None:
        if not self.rate_limiter.check(ctx.author.id, "fun"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "fun")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        target = target or ctx.author

        async with ctx.typing():
            response = await self.ai.generate(
                messages=[{"role": "user", "content": f"Roast {target.display_name} in a funny, playful way. Keep it lighthearted and clever, not mean-spirited."}],
                system="You are a witty roast comedian. Be funny and clever, but always good-natured. Never cross into genuinely hurtful territory.",
                max_tokens=300,
                temperature=1.0,
            )
            self.rate_limiter.record(ctx.author.id, "fun")

        embed = discord.Embed(
            title=f"Roasting {target.display_name}",
            description=response.content,
            color=discord.Color.red(),
        )
        await ctx.reply(target.mention, embed=embed, mention_author=False)

    @commands.hybrid_command(name="compliment")
    @app_commands.describe(target="Who to compliment")
    async def compliment_user(self, ctx: commands.Context, target: discord.Member | None = None) -> None:
        if not self.rate_limiter.check(ctx.author.id, "fun"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "fun")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        target = target or ctx.author

        async with ctx.typing():
            response = await self.ai.generate(
                messages=[{"role": "user", "content": f"Give {target.display_name} a creative, heartfelt compliment. Make it unique and genuine."}],
                system="You are a positivity expert. Give genuine, creative compliments that make people smile.",
                max_tokens=200,
                temperature=0.9,
            )
            self.rate_limiter.record(ctx.author.id, "fun")

        embed = discord.Embed(
            title=f"Compliment for {target.display_name}",
            description=response.content,
            color=discord.Color.green(),
        )
        await ctx.reply(target.mention, embed=embed, mention_author=False)

    @commands.hybrid_command(name="ask_anything")
    @app_commands.describe(question="Your question")
    async def ask_anything(self, ctx: commands.Context, *, question: str) -> None:
        if not self.rate_limiter.check(ctx.author.id, "fun"):
            info = self.rate_limiter.get_usage_info(ctx.author.id, "fun")
            await ctx.reply(f"Rate limit reached. Wait {info['wait_seconds']}s.", mention_author=False)
            return

        async with ctx.typing():
            response = await self.ai.generate(
                messages=[{"role": "user", "content": question}],
                system="You are a knowledgeable, friendly AI. Answer any question thoroughly but concisely.",
            )
            self.rate_limiter.record(ctx.author.id, "fun")

        if len(response.content) > 2000:
            file = discord.File(
                fp=__import__("io").BytesIO(response.content.encode()),
                filename="answer.md",
            )
            await ctx.reply("Answer is long. See attached file:", file=file, mention_author=False)
        else:
            embed = discord.Embed(description=response.content, color=discord.Color.blue())
            await ctx.reply(embed=embed, mention_author=False)

    @app_commands.command(name="personas", description="List all available AI personas")
    async def list_personas(self, interaction: discord.Interaction) -> None:
        embed = discord.Embed(title="AI Personas", color=discord.Color.purple())
        for name, desc in PERSONAS.items():
            short_desc = desc[:80] + "..." if len(desc) > 80 else desc
            embed.add_field(name=name.title(), value=short_desc, inline=False)
        embed.set_footer(text="Use /persona <name> <message> to chat with a persona")
        await interaction.response.send_message(embed=embed, ephemeral=True)
