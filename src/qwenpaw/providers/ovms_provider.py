# -*- coding: utf-8 -*-
"""An OVMS provider implementation."""

from typing import Any

from agentscope.model import ChatModelBase
import httpx
from openai import AsyncOpenAI

from qwenpaw.providers.openai_provider import OpenAIProvider


class OVMSChatModelCompat:
    """Mixin for OVMS-compatible OpenAI chat calls."""

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        sanitized_kwargs = {
            key: value for key, value in kwargs.items() if value is not None
        }
        return await super().__call__(*args, **sanitized_kwargs)  # type: ignore[misc]


class OVMSProvider(OpenAIProvider):
    """Provider implementation for OpenVINO Model Server."""

    def get_effective_generate_kwargs(self, model_id: str) -> dict[str, Any]:
        generate_kwargs = super().get_effective_generate_kwargs(model_id)
        sanitized_kwargs = {
            key: value
            for key, value in generate_kwargs.items()
            if value is not None
        }
        sanitized_kwargs.setdefault("max_tokens", 1024)
        return sanitized_kwargs

    def _client(self, timeout: float = 5) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            http_client=httpx.AsyncClient(trust_env=False),
        )

    def get_chat_model_instance(self, model_id: str) -> ChatModelBase:
        from .openai_chat_model_compat import OpenAIChatModelCompat

        class OVMSOpenAIChatModelCompat(
            OVMSChatModelCompat,
            OpenAIChatModelCompat,
        ):
            pass

        return OVMSOpenAIChatModelCompat(
            model_name=model_id,
            stream=True,
            api_key=self.api_key,
            stream_tool_parsing=False,
            client_kwargs={
                "base_url": self.base_url,
                "http_client": httpx.AsyncClient(trust_env=False),
            },
            generate_kwargs=self.get_effective_generate_kwargs(model_id),
        )