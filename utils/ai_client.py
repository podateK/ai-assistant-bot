from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx
from loguru import logger
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import AIProvider, Settings


@dataclass
class AIResponse:
    content: str
    model: str
    provider: AIProvider
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageResponse:
    url: str
    revised_prompt: str = ""
    model: str = ""
    provider: AIProvider = AIProvider.OPENAI


class AIClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._openai: AsyncOpenAI | None = None
        self._anthropic: anthropic.AsyncAnthropic | None = None
        self._http: httpx.AsyncClient | None = None
        self._fallback_order = self._build_fallback_order()

    def _build_fallback_order(self) -> list[AIProvider]:
        providers = []
        if self._settings.openai_api_key:
            providers.append(AIProvider.OPENAI)
        if self._settings.anthropic_api_key:
            providers.append(AIProvider.ANTHROPIC)
        if self._settings.local_api_url:
            providers.append(AIProvider.LOCAL)
        return providers

    async def initialize(self) -> None:
        if self._settings.openai_api_key:
            self._openai = AsyncOpenAI(
                api_key=self._settings.openai_api_key,
                organization=self._settings.openai_org_id or None,
                timeout=httpx.Timeout(120.0, connect=30.0),
            )
        if self._settings.anthropic_api_key:
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=self._settings.anthropic_api_key,
                timeout=httpx.Timeout(120.0, connect=30.0),
            )
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0))
        logger.info("AI client initialized with providers: {}", [p.value for p in self._fallback_order])

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()

    def get_model(self, provider: AIProvider | None = None) -> str:
        target = provider or self._settings.default_ai_model
        match target:
            case AIProvider.OPENAI | "openai":
                return self._settings.openai_model
            case AIProvider.ANTHROPIC | "anthropic":
                return self._settings.anthropic_model
            case AIProvider.LOCAL | "local":
                return self._settings.local_model
            case _:
                return self._settings.openai_model

    def _select_provider(self, preferred: str | AIProvider | None = None) -> AIProvider:
        if preferred:
            provider = AIProvider(preferred) if isinstance(preferred, str) else preferred
            if provider in self._fallback_order:
                return provider
        default = AIProvider(self._settings.default_ai_model)
        if default in self._fallback_order:
            return default
        return self._fallback_order[0] if self._fallback_order else AIProvider.OPENAI

    @retry(
        retry=retry_if_exception_type((anthropic.APIError, anthropic.RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_openai(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        if not self._openai:
            raise ValueError("OpenAI client not initialized")

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = await self._openai.chat.completions.create(
            model=model or self._settings.openai_model,
            messages=full_messages,
            max_tokens=max_tokens or self._settings.openai_max_tokens,
            temperature=temperature if temperature is not None else self._settings.openai_temperature,
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage
        return AIResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=AIProvider.OPENAI,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "",
        )

    @retry(
        retry=retry_if_exception_type((anthropic.APIError, anthropic.RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_anthropic(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        if not self._anthropic:
            raise ValueError("Anthropic client not initialized")

        call_kwargs: dict[str, Any] = {
            "model": model or self._settings.anthropic_model,
            "messages": messages,
            "max_tokens": max_tokens or self._settings.anthropic_max_tokens,
            "temperature": temperature if temperature is not None else self._settings.anthropic_temperature,
        }
        if system:
            call_kwargs["system"] = system

        response = await self._anthropic.messages.create(**call_kwargs)
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return AIResponse(
            content=content,
            model=response.model,
            provider=AIProvider.ANTHROPIC,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason or "",
        )

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_local(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        if not self._http:
            raise ValueError("HTTP client not initialized")

        prompt_parts = []
        if system:
            prompt_parts.append(f"System: {system}")
        for msg in messages:
            role = msg["role"].capitalize()
            prompt_parts.append(f"{role}: {msg['content']}")
        prompt_parts.append("Assistant:")

        payload: dict[str, Any] = {
            "model": model or self._settings.local_model,
            "prompt": "\n".join(prompt_parts),
            "stream": False,
            "options": {
                "num_predict": max_tokens or 2048,
                "temperature": temperature if temperature is not None else 0.7,
            },
        }
        if self._settings.local_api_key:
            payload["key"] = self._settings.local_api_key

        response = await self._http.post(self._settings.local_api_url, json=payload)
        response.raise_for_status()
        data = response.json()

        return AIResponse(
            content=data.get("response", ""),
            model=data.get("model", self._settings.local_model),
            provider=AIProvider.LOCAL,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        provider: str | AIProvider | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        target = self._select_provider(provider)
        call_map = {
            AIProvider.OPENAI: self._call_openai,
            AIProvider.ANTHROPIC: self._call_anthropic,
            AIProvider.LOCAL: self._call_local,
        }

        last_error = None
        ordered_providers = [target] + [p for p in self._fallback_order if p != target]

        for prov in ordered_providers:
            if prov not in call_map:
                continue
            try:
                response = await call_map[prov](
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    **kwargs,
                )
                return response
            except Exception as e:
                last_error = e
                logger.warning("Provider {} failed: {}", prov.value, e)
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    async def generate_image(
        self,
        prompt: str,
        size: str | None = None,
        quality: str | None = None,
        model: str | None = None,
    ) -> ImageResponse:
        if not self._openai:
            raise ValueError("OpenAI client required for image generation")

        response = await self._openai.images.generate(
            model=model or self._settings.dall_e_model,
            prompt=prompt,
            size=size or self._settings.dall_e_size,
            quality=quality or self._settings.dall_e_quality,
            n=1,
            response_format="url",
        )

        image_data = response.data[0]
        return ImageResponse(
            url=image_data.url or "",
            revised_prompt=getattr(image_data, "revised_prompt", "") or "",
            model=model or self._settings.dall_e_model,
            provider=AIProvider.OPENAI,
        )

    async def stream_generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        provider: str | AIProvider | None = None,
    ):
        target = self._select_provider(provider)

        if target == AIProvider.OPENAI and self._openai:
            full_messages = []
            if system:
                full_messages.append({"role": "system", "content": system})
            full_messages.extend(messages)
            stream = await self._openai.chat.completions.create(
                model=model or self._settings.openai_model,
                messages=full_messages,
                max_tokens=max_tokens or self._settings.openai_max_tokens,
                temperature=temperature if temperature is not None else self._settings.openai_temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        elif target == AIProvider.ANTHROPIC and self._anthropic:
            call_kwargs: dict[str, Any] = {
                "model": model or self._settings.anthropic_model,
                "messages": messages,
                "max_tokens": max_tokens or self._settings.anthropic_max_tokens,
                "temperature": temperature if temperature is not None else self._settings.anthropic_temperature,
            }
            if system:
                call_kwargs["system"] = system
            async with self._anthropic.messages.stream(**call_kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        else:
            response = await self.generate(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                provider=provider,
            )
            yield response.content
