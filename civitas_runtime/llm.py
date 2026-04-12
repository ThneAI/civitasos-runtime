"""LLM Adapter — unified interface for OpenAI / Anthropic / LiteLLM backends.

Injects CivitasOS context as system prompt automatically.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from .models import LLMResponse, ToolCall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract protocol
# ---------------------------------------------------------------------------

class LLMAdapter(ABC):
    """Interface every LLM backend must implement."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a chat completion request. Returns unified LLMResponse."""
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter (covers OpenAI, Azure OpenAI, vLLM, Ollama, etc.)
# ---------------------------------------------------------------------------

class OpenAIAdapter(LLMAdapter):
    """Adapter for any OpenAI-compatible chat/completions endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Install openai: pip install openai"
            ) from exc
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = await self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        )


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter(LLMAdapter):
    """Adapter for the Anthropic Messages API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ) -> None:
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Install anthropic: pip install anthropic"
            ) from exc
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = AsyncAnthropic(**kwargs)
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        # Separate system message from user/assistant messages
        system_text = ""
        chat_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                chat_msgs.append(m)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": chat_msgs,
            "max_tokens": 4096,
            "temperature": temperature,
        }
        if system_text:
            kwargs["system"] = system_text.strip()

        # Convert OpenAI tool format to Anthropic format
        if tools:
            anthropic_tools = []
            for t in tools:
                fn = t.get("function", t)
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            kwargs["tools"] = anthropic_tools

        resp = await self._client.messages.create(**kwargs)

        content_text = ""
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return LLMResponse(
            content=content_text or None,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            },
        )


# ---------------------------------------------------------------------------
# LiteLLM adapter (100+ models)
# ---------------------------------------------------------------------------

class LiteLLMAdapter(LLMAdapter):
    """Adapter using litellm for 100+ model providers."""

    def __init__(self, model: str = "gpt-4o") -> None:
        try:
            import litellm  # type: ignore[import-untyped] # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install litellm: pip install litellm"
            ) from exc
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        import litellm  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = await litellm.acompletion(**kwargs)
        msg = resp.choices[0].message

        tool_calls: list[ToolCall] = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str)
                    else tc.function.arguments,
                ))

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            usage=dict(resp.usage) if resp.usage else {},
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm(spec: str, **kwargs: Any) -> LLMAdapter:
    """Create an LLM adapter from a spec string like 'openai:gpt-4o'.

    Supported prefixes: openai, anthropic, litellm.
    If no prefix, defaults to openai.
    """
    if ":" in spec:
        provider, model = spec.split(":", 1)
    else:
        provider, model = "openai", spec

    provider = provider.lower()
    if provider == "openai":
        return OpenAIAdapter(model=model, **kwargs)
    elif provider == "anthropic":
        return AnthropicAdapter(model=model, **kwargs)
    elif provider == "litellm":
        return LiteLLMAdapter(model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}. Use openai/anthropic/litellm.")
