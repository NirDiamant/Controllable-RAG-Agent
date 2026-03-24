"""
Multi-provider LLM factory for the Controllable RAG Agent.

Supports OpenAI (default) and MiniMax via environment variables.

Usage:
    Set LLM_PROVIDER=minimax and MINIMAX_API_KEY=... in your .env to use MiniMax.
    Defaults to OpenAI when LLM_PROVIDER is unset or set to "openai".
"""

import re
import os
from typing import Any, List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


# MiniMax API configuration
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_DEFAULT_MODEL = "MiniMax-M2.5"
MINIMAX_MODELS = ["MiniMax-M2.5", "MiniMax-M2.5-highspeed"]

# Regex to strip <think>...</think> tags from MiniMax responses
_THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from MiniMax model output."""
    return _THINK_TAG_RE.sub("", text).strip()


class ChatMiniMax(ChatOpenAI):
    """ChatOpenAI subclass that strips <think> tags from MiniMax responses.

    MiniMax M2.5 models may include <think>...</think> reasoning blocks
    in their output. This wrapper transparently removes them so that
    downstream parsers (e.g. structured output / JSON mode) work correctly.
    """

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        result = super()._generate(messages, stop=stop, **kwargs)
        for gen in result.generations:
            if hasattr(gen, "message") and hasattr(gen.message, "content"):
                gen.message.content = _strip_think_tags(gen.message.content)
            if hasattr(gen, "text"):
                gen.text = _strip_think_tags(gen.text)
        return result


def _get_provider() -> str:
    """Return the configured LLM provider name, lowercased."""
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def _clamp_temperature(temperature: float) -> float:
    """MiniMax requires temperature in (0.0, 1.0]. Clamp 0 to a small epsilon."""
    if temperature <= 0:
        return 0.01
    if temperature > 1.0:
        return 1.0
    return temperature


def get_chat_llm(temperature: float = 0, model_name: str = "gpt-4o", max_tokens: int = 2000) -> ChatOpenAI:
    """
    Create a ChatOpenAI-compatible LLM instance for the configured provider.

    For MiniMax, routes through the OpenAI-compatible endpoint at api.minimax.io.

    Args:
        temperature: Sampling temperature.
        model_name: Model name (used for OpenAI; overridden for MiniMax).
        max_tokens: Maximum tokens to generate.

    Returns:
        A ChatOpenAI instance configured for the selected provider.
    """
    provider = _get_provider()

    if provider == "minimax":
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("MINIMAX_API_KEY environment variable is required when LLM_PROVIDER=minimax")

        minimax_model = os.getenv("MINIMAX_MODEL", MINIMAX_DEFAULT_MODEL)
        clamped_temp = _clamp_temperature(temperature)

        return ChatMiniMax(
            temperature=clamped_temp,
            model=minimax_model,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            openai_api_base=MINIMAX_BASE_URL,
        )

    # Default: OpenAI
    return ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=max_tokens)


def get_embeddings() -> OpenAIEmbeddings:
    """
    Create an embeddings instance for the configured provider.

    Note: MiniMax's native embedding API (embo-01) is not OpenAI-compatible,
    so this falls back to OpenAI embeddings regardless of provider.
    Users who need MiniMax embeddings should use the MiniMax API directly.

    Returns:
        An OpenAIEmbeddings instance.
    """
    return OpenAIEmbeddings()
