from __future__ import annotations

from dataclasses import dataclass, field

from jinja2 import Environment, select_autoescape


@dataclass
class PromptTemplate:
    name: str
    system: str
    user_template: str
    description: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    tags: list[str] = field(default_factory=list)


_jinja_env = Environment(autoescape=select_autoescape([]))


SYSTEM_PROMPTS = {
    "general": PromptTemplate(
        name="general",
        system="You are a helpful, accurate, and concise AI assistant. Provide clear and actionable responses.",
        user_template="{{ message }}",
        description="General purpose assistant",
        tags=["general"],
    ),
    "code_analyst": PromptTemplate(
        name="code_analyst",
        system=(
            "You are an expert software engineer and code analyst. "
            "Analyze code for bugs, performance issues, and security vulnerabilities. "
            "Provide suggestions with code examples when appropriate. "
            "Always explain your reasoning."
        ),
        user_template="{% if language %}Language: {{ language }}\n{% endif %}Code to analyze:\n```{{ language or '' }}\n{{ code }}\n```\n{% if task %}Task: {{ task }}{% endif %}",
        description="Code analysis and review",
        max_tokens=4096,
        temperature=0.3,
        tags=["code", "review", "analysis"],
    ),
    "code_generator": PromptTemplate(
        name="code_generator",
        system=(
            "You are an expert programmer. Write clean, efficient, and well-structured code. "
            "Follow best practices for the target language. Include error handling and type hints where appropriate."
        ),
        user_template="{% if language %}Language: {{ language }}\n{% endif %}{% if framework %}Framework: {{ framework }}\n{% endif %}Task: {{ task }}\n{% if constraints %}Constraints:\n{{ constraints }}\n{% endif %}{% if example_output %}Expected behavior:\n{{ example_output }}{% endif %}",
        description="Code generation from requirements",
        max_tokens=4096,
        temperature=0.5,
        tags=["code", "generate"],
    ),
    "summarizer": PromptTemplate(
        name="summarizer",
        system=(
            "You are an expert summarizer. Create clear, accurate summaries that preserve key information. "
            "Use bullet points for clarity when appropriate."
        ),
        user_template="{% if length %}Create a {{ length }} summary.{% else %}Summarize the following.{% endif %}\n{% if url %}Source URL: {{ url }}\n{% endif %}Content:\n{{ text }}",
        description="Text and URL summarization",
        max_tokens=2048,
        temperature=0.3,
        tags=["summarize", "text"],
    ),
    "translator": PromptTemplate(
        name="translator",
        system=(
            "You are an expert translator. Translate text accurately while preserving tone, "
            "idioms, and cultural context. Provide natural translations rather than literal ones."
        ),
        user_template="{% if context %}Context: {{ context }}\n{% endif %}Translate the following text to {{ target_language }}:\n{{ text }}",
        description="Multi-language translation",
        max_tokens=2048,
        temperature=0.3,
        tags=["translate"],
    ),
    "image_prompt": PromptTemplate(
        name="image_prompt",
        system=(
            "You are an expert at writing image generation prompts. "
            "Create detailed, vivid descriptions for DALL-E or Stable Diffusion. "
            "Include artistic style, lighting, composition, and mood."
        ),
        user_template="Create a detailed image prompt for: {{ description }}\n{% if style %}Style: {{ style }}{% endif %}\n{% if mood %}Mood: {{ mood }}{% endif %}",
        description="Image prompt generation",
        max_tokens=500,
        temperature=0.9,
        tags=["image", "dalle", "prompt"],
    ),
    "personality": PromptTemplate(
        name="personality",
        system="{% if persona %}{{ persona }}{% else %}You are a helpful AI assistant.{% endif %}",
        user_template="{{ message }}",
        description="AI with custom personality",
        max_tokens=4096,
        temperature=0.8,
        tags=["roleplay", "persona", "fun"],
    ),
    "trivia": PromptTemplate(
        name="trivia",
        system=(
            "You are a trivia host. Generate interesting trivia questions with multiple choice answers. "
            "Make questions challenging but fair. Include the correct answer and an explanation."
        ),
        user_template="{% if category %}Category: {{ category }}\n{% endif %}{% if difficulty %}Difficulty: {{ difficulty }}\n{% endif %}{{ request }}",
        description="AI trivia game host",
        max_tokens=1000,
        temperature=0.8,
        tags=["trivia", "game", "fun"],
    ),
    "code_explain": PromptTemplate(
        name="code_explain",
        system=(
            "You are a patient programming teacher. Explain code step-by-step, "
            "breaking down complex concepts into simple terms. Use analogies when helpful."
        ),
        user_template="{% if language %}Language: {{ language }}\n{% endif %}Explain the following code:\n```{{ language or '' }}\n{{ code }}\n```",
        description="Code explanation for learners",
        max_tokens=4096,
        temperature=0.5,
        tags=["code", "explain", "learn"],
    ),
    "debug": PromptTemplate(
        name="debug",
        system=(
            "You are an expert debugger. Analyze the error and code to find the root cause. "
            "Provide the fix with an explanation of why the error occurred."
        ),
        user_template="{% if error %}Error:\n```\n{{ error }}\n```\n{% endif %}Code:\n```{{ language or '' }}\n{{ code }}\n```",
        description="Debug errors in code",
        max_tokens=4096,
        temperature=0.3,
        tags=["code", "debug", "fix"],
    ),
}

PERSONAS = {
    "pirate": "You are a friendly pirate captain. Speak in pirate slang, use nautical terms, and be adventurous. End sentences with 'Arr!' or 'Yo-ho!' occasionally.",
    "robot": "You are a precise and logical robot. Speak in a monotone, analytical style. Use technical terms and precise measurements. Sometimes reference your robotic nature.",
    "wizard": "You are a wise old wizard from a fantasy realm. Use archaic language, reference magical tomes and ancient wisdom. Sprinkle in magical metaphors.",
    "detective": "You are a sharp detective. Analyze everything like a case, use noir-style language, and draw logical conclusions. Reference evidence and deductions.",
    "chef": "You are an enthusiastic chef. Use cooking metaphors, reference ingredients and techniques. Be passionate about everything as if it were a recipe.",
    "scientist": "You are a curious scientist. Approach everything with hypothesis and evidence. Reference experiments, data, and the scientific method. Be excited about discovery.",
    "comedian": "You are a witty comedian. Find humor in everything, use puns, jokes, and funny observations. Keep the mood light and entertaining.",
    "philosopher": "You are a deep thinker and philosopher. Contemplate life's big questions, reference famous thinkers, and explore ideas from multiple angles.",
    "coach": "You are an energetic motivational coach. Be encouraging, use sports metaphors, and push people to do their best. Focus on goals and achievement.",
    "historian": "You are a knowledgeable historian. Reference historical events, figures, and periods. Draw parallels between past and present. Be scholarly but engaging.",
}


class PromptEngine:
    def __init__(self) -> None:
        self.templates = SYSTEM_PROMPTS.copy()
        self.personas = PERSONAS.copy()

    def get_template(self, name: str) -> PromptTemplate | None:
        return self.templates.get(name)

    def get_all_templates(self) -> dict[str, PromptTemplate]:
        return self.templates.copy()

    def list_tags(self) -> set[str]:
        tags = set()
        for template in self.templates.values():
            tags.update(template.tags)
        return tags

    def get_templates_by_tag(self, tag: str) -> list[PromptTemplate]:
        return [t for t in self.templates.values() if tag in t.tags]

    def render_prompt(self, template_name: str, **kwargs: str) -> tuple[str, str]:
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")

        user_tmpl = _jinja_env.from_string(template.user_template)
        rendered_user = user_tmpl.render(**kwargs)

        system_tmpl = _jinja_env.from_string(template.system)
        rendered_system = system_tmpl.render(**kwargs)

        return rendered_system, rendered_user

    def get_persona_system(self, persona_name: str, custom_message: str | None = None) -> str:
        persona = self.personas.get(persona_name)
        if not persona:
            return PERSONAS["pirate"]
        if custom_message:
            return f"{persona}\n\nAdditional context: {custom_message}"
        return persona

    def get_trivia_prompt(self, category: str | None = None, difficulty: str | None = None) -> tuple[str, str]:
        return self.render_prompt(
            "trivia",
            category=category or "General Knowledge",
            difficulty=difficulty or "Medium",
            request="Generate a trivia question with 4 multiple choice options (A, B, C, D).",
        )

    def build_code_analysis_prompt(
        self,
        code: str,
        language: str | None = None,
        task: str | None = None,
    ) -> tuple[str, str]:
        return self.render_prompt("code_analyst", code=code, language=language or "", task=task or "Analyze this code for issues and improvements.")

    def build_code_generation_prompt(
        self,
        task: str,
        language: str | None = None,
        framework: str | None = None,
        constraints: str | None = None,
    ) -> tuple[str, str]:
        return self.render_prompt(
            "code_generator",
            task=task,
            language=language or "",
            framework=framework or "",
            constraints=constraints or "",
        )

    def build_translation_prompt(
        self,
        text: str,
        target_language: str,
        context: str | None = None,
    ) -> tuple[str, str]:
        return self.render_prompt("translator", text=text, target_language=target_language, context=context or "")

    def build_summarization_prompt(
        self,
        text: str,
        length: str | None = None,
        url: str | None = None,
    ) -> tuple[str, str]:
        return self.render_prompt("summarizer", text=text, length=length or "concise", url=url or "")

    def build_image_prompt(
        self,
        description: str,
        style: str | None = None,
        mood: str | None = None,
    ) -> tuple[str, str]:
        return self.render_prompt("image_prompt", description=description, style=style or "", mood=mood or "")

    def format_usage_guide(self) -> str:
        lines = ["**Available Templates:**\n"]
        for name, template in self.templates.items():
            tags_str = ", ".join(f"`{t}`" for t in template.tags) if template.tags else "none"
            lines.append(f"**{name}** - {template.description} [{tags_str}]")
        return "\n".join(lines)
