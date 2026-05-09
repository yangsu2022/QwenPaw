# -*- coding: utf-8 -*-
"""Direct Tavily search tool (HTTP API, no MCP dependency)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse


def _read_tavily_key_from_agent_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    mcp = data.get("mcp") if isinstance(data, dict) else None
    clients = mcp.get("clients") if isinstance(mcp, dict) else None
    tavily = clients.get("tavily_search") if isinstance(clients, dict) else None
    env = tavily.get("env") if isinstance(tavily, dict) else None
    key = env.get("TAVILY_API_KEY") if isinstance(env, dict) else ""
    return key if isinstance(key, str) else ""


def _resolve_tavily_api_key() -> str:
    env_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if env_key:
        return env_key

    cwd_agent_json = Path.cwd() / "agent.json"
    if cwd_agent_json.exists():
        key = _read_tavily_key_from_agent_json(cwd_agent_json).strip()
        if key:
            return key

    default_agent_json = Path.home() / ".qwenpaw" / "workspaces" / "default" / "agent.json"
    if default_agent_json.exists():
        key = _read_tavily_key_from_agent_json(default_agent_json).strip()
        if key:
            return key

    return ""


async def tavily_search(
    query: str,
    max_results: int = 5,
    topic: str = "news",
) -> ToolResponse:
    """Search the web via Tavily Search API.

    Use this tool for online search requests when current/fresh information is needed.

    Args:
        query: Search query text.
        max_results: Max number of results to return (1-10).
        topic: Search topic (e.g. "news" or "general").

    Returns:
        ToolResponse with compact JSON result list.
    """
    q = (query or "").strip()
    if not q:
        return ToolResponse(
            content=[TextBlock(type="text", text="Error: query is empty.")],
        )

    key = _resolve_tavily_api_key()
    if not key:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Tavily API key missing. Set TAVILY_API_KEY or configure mcp.clients.tavily_search.env.TAVILY_API_KEY in agent.json.",
                ),
            ],
        )

    n = max(1, min(int(max_results), 10))
    payload: dict[str, Any] = {
        "query": q,
        "max_results": n,
        "topic": topic,
        "api_key": key,
    }

    try:
        # trust_env=True: keep enterprise proxy compatibility when required.
        async with httpx.AsyncClient(timeout=30, trust_env=True) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return ToolResponse(
            content=[TextBlock(type="text", text=f"Tavily API error: {exc}")],
        )

    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        results = []

    compact = []
    for item in results[:n]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            },
        )

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(
                    {
                        "query": q,
                        "max_results": n,
                        "results": compact,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
        ],
    )
