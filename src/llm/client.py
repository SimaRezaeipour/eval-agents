"""Unified LLM client — uses Anthropic if ANTHROPIC_API_KEY is set, else Azure OpenAI."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.settings import get_settings

logger = logging.getLogger(__name__)


def complete(system: str, user: str, max_tokens: int = 2048) -> str:
    """Send a system+user prompt and return the response text as a string.

    Automatically selects Anthropic, Azure OpenAI, Google Gemini, or a generic OpenAI-compatible
    backend based on which credentials are configured. All backends are instructed to return JSON only.
    """
    settings = get_settings()

    if settings.anthropic_api_key.get_secret_value():
        return _anthropic_complete(system, user, max_tokens, settings)
    elif settings.azure_openai_key.get_secret_value() and settings.azure_openai_endpoint:
        return _azure_complete(system, user, max_tokens, settings)
    elif settings.openai_api_key.get_secret_value() and settings.openai_base_url:
        return _openai_complete(system, user, max_tokens, settings)
    elif settings.openai_api_key.get_secret_value():
        return _gemini_complete(system, user, max_tokens, settings)
    else:
        raise RuntimeError(
            "No LLM credentials found. Set ANTHROPIC_API_KEY in .env, "
            "AZURE_OPENAI_KEY + AZURE_OPENAI_ENDPOINT for Azure OpenAI, "
            "a Google API key via OPENAI_API_KEY with the Gemini-compatible setup, or OPENAI_API_KEY."
        )


def _anthropic_complete(system: str, user: str, max_tokens: int, settings: Any) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        system=system + "\n\nIMPORTANT: Respond with valid JSON only. No markdown, no code fences.",
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _azure_complete(system: str, user: str, max_tokens: int, settings: Any) -> str:
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=settings.azure_openai_key.get_secret_value(),
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    response = client.chat.completions.create(
        model=settings.azure_openai_deployment,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or "{}"


def _openai_complete(system: str, user: str, max_tokens: int, settings: Any) -> str:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {
        "api_key": settings.openai_api_key.get_secret_value(),
    }
    if settings.openai_base_url:
        client_kwargs["base_url"] = settings.openai_base_url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or "{}"


def _gemini_complete(system: str, user: str, max_tokens: int, settings: Any) -> str:
    import google.generativeai as genai

    genai.configure(api_key=settings.openai_api_key.get_secret_value())
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    response = model.generate_content(
        [
            {"role": "user", "parts": [f"{system}\n\nIMPORTANT: Respond with valid JSON only. No markdown, no code fences.\n\n{user}"]},
        ],
        generation_config={"temperature": 0, "max_output_tokens": max_tokens},
    )
    return response.text or "{}"
