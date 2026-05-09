# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio

from qwenpaw.providers.ovms_provider import OVMSProvider


def _make_provider() -> OVMSProvider:
    return OVMSProvider(
        id="ovms",
        name="OpenVINO Model Server",
        base_url="http://localhost:8000/v3",
        api_key="",
        chat_model="OpenAIChatModel",
    )


def test_client_bypasses_environment_proxies(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            captured["http_client_kwargs"] = kwargs

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("qwenpaw.providers.ovms_provider.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("qwenpaw.providers.ovms_provider.AsyncOpenAI", FakeAsyncOpenAI)

    provider = _make_provider()
    getattr(provider, "_client")(timeout=7)

    assert captured["base_url"] == "http://localhost:8000/v3"
    assert captured["api_key"] == ""
    assert captured["timeout"] == 7
    assert captured["http_client_kwargs"] == {"trust_env": False}


def test_get_chat_model_instance_bypasses_environment_proxies(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            captured["http_client_kwargs"] = kwargs

    class FakeOpenAIChatModelCompat:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("qwenpaw.providers.ovms_provider.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        "qwenpaw.providers.openai_chat_model_compat.OpenAIChatModelCompat",
        FakeOpenAIChatModelCompat,
    )

    provider = _make_provider()
    provider.get_chat_model_instance("OpenVINO/Qwen3-8B-int4-ov")

    assert captured["model_name"] == "OpenVINO/Qwen3-8B-int4-ov"
    assert captured["api_key"] == ""
    assert captured["stream"] is True
    assert captured["stream_tool_parsing"] is False
    assert captured["client_kwargs"]["base_url"] == "http://localhost:8000/v3"
    assert captured["client_kwargs"]["http_client"] is not None
    assert captured["http_client_kwargs"] == {"trust_env": False}
    assert captured["generate_kwargs"] == {"max_tokens": 1024}


def test_effective_generate_kwargs_replaces_saved_null_max_tokens() -> None:
    provider = OVMSProvider(
        id="ovms",
        name="OpenVINO Model Server",
        base_url="http://localhost:8000/v3",
        api_key="",
        chat_model="OpenAIChatModel",
        generate_kwargs={"max_tokens": None, "temperature": None},
    )

    assert provider.get_effective_generate_kwargs(
        "OpenVINO/Qwen3-8B-int4-ov",
    ) == {"max_tokens": 1024}


def test_chat_model_omits_none_request_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return object()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    provider = OVMSProvider(
        id="ovms",
        name="OpenVINO Model Server",
        base_url="http://localhost:8000/v3",
        api_key="",
        chat_model="OpenAIChatModel",
        generate_kwargs={"max_tokens": 1024},
    )
    model = provider.get_chat_model_instance("OpenVINO/Qwen3-8B-int4-ov")
    model.client = FakeClient()

    asyncio.run(
        model(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=None,
            temperature=None,
        ),
    )

    assert captured["model"] == "OpenVINO/Qwen3-8B-int4-ov"
    assert captured["messages"] == [{"role": "user", "content": "ping"}]
    assert captured["stream"] is True
    assert captured["max_tokens"] == 1024
    assert "temperature" not in captured