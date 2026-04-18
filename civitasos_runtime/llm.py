"""LLM Adapter — unified interface for OpenAI / Anthropic / LiteLLM backends.

Injects CivitasOS context as system prompt automatically.
Includes retry with exponential backoff, request timeouts, and graceful
fallback so one transient LLM failure doesn't crash the cognitive loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from .models import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

# Default retry / timeout values
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_BASE_DELAY = 1.0  # exponential backoff base (seconds)


# ---------------------------------------------------------------------------
# Abstract protocol
# ---------------------------------------------------------------------------

class LLMAdapter(ABC):
    """Interface every LLM backend must implement."""

    def __init__(
        self,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        base_delay: float = DEFAULT_BASE_DELAY,
    ) -> None:
        self._max_retries = max_retries
        self._timeout = timeout
        self._base_delay = base_delay

    # -- public API (with retry / timeout) ----------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send chat completion with retry + timeout. Falls back on total failure."""
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._do_chat(messages, tools=tools, temperature=temperature),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"LLM request timed out after {self._timeout}s"
                )
                logger.warning(
                    "LLM timeout (attempt %d/%d, %.0fs)",
                    attempt, self._max_retries, self._timeout,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM error (attempt %d/%d): %s",
                    attempt, self._max_retries, exc,
                )
            if attempt < self._max_retries:
                delay = self._base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)

        # All retries exhausted — return a safe fallback response
        logger.error("LLM failed after %d attempts: %s", self._max_retries, last_error)
        return LLMResponse(
            content="wait",
            tool_calls=[],
            usage={},
        )

    # -- subclass contract ---------------------------------------------------

    @abstractmethod
    async def _do_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Raw chat completion without retry/timeout. Subclasses implement this."""
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
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        base_delay: float = DEFAULT_BASE_DELAY,
        disable_thinking: bool | None = None,
    ) -> None:
        super().__init__(max_retries=max_retries, timeout=timeout, base_delay=base_delay)
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
        # Auto-detect: opt-in via env, or default-on for known thinking models
        # served via Ollama (qwen3, deepseek-r1, qwq, ...) where the long
        # <think>...</think> chain dominates wall-clock for benchmarks.
        if disable_thinking is None:
            env = os.environ.get("LLM_DISABLE_THINKING", "").strip().lower()
            if env in ("1", "true", "yes"):
                disable_thinking = True
            elif env in ("0", "false", "no"):
                disable_thinking = False
            else:
                lower = model.lower()
                is_ollama = bool(base_url and "11434" in base_url)
                disable_thinking = is_ollama and any(
                    tag in lower for tag in ("qwen3", "deepseek-r1", "qwq")
                )
        self._disable_thinking = disable_thinking

    async def _do_chat(
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
        if self._disable_thinking:
            # Two redundant switches because Ollama versions disagree:
            #  1. extra_body.chat_template_kwargs.enable_thinking — newer Ollama
            #     forwards this to the qwen3 jinja template.
            #  2. Append qwen3's native "/no_think" tag to the system message —
            #     understood by the model itself regardless of Ollama version.
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
            messages = _inject_no_think(messages)
            kwargs["messages"] = messages

        resp = await self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        # Strip residual <think>...</think> blocks if the model emitted them
        # despite the hint (some Ollama builds ignore extra_body and the
        # /no_think tag is non-binding).
        content = msg.content
        if self._disable_thinking and content and "<think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return LLMResponse(
            content=content,
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
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        base_delay: float = DEFAULT_BASE_DELAY,
    ) -> None:
        super().__init__(max_retries=max_retries, timeout=timeout, base_delay=base_delay)
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

    async def _do_chat(
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

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        base_delay: float = DEFAULT_BASE_DELAY,
    ) -> None:
        super().__init__(max_retries=max_retries, timeout=timeout, base_delay=base_delay)
        try:
            import litellm  # type: ignore[import-untyped] # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install litellm: pip install litellm"
            ) from exc
        self._model = model

    async def _do_chat(
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

def _inject_no_think(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append qwen3's "/no_think" directive to the system message.

    Returns a shallow copy with the system message rewritten; the original
    list is not mutated. If no system message exists, prepends one.
    """
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            content = m.get("content") or ""
            if "/no_think" in content:
                return out
            out[i] = {**m, "content": content.rstrip() + " /no_think"}
            return out
    return [{"role": "system", "content": "/no_think"}, *out]


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
