"""Deployable research agent used by Modules 3 and 5.

Demonstrates:
- AGENTS.md for agent identity and instructions
- Skills for on-demand capabilities (LinkedIn, Twitter)
- Custom tools (Tavily search)
- Research subagent for delegated work
- CompositeBackend: FilesystemBackend for skills/AGENTS.md, StoreBackend for /memories/
- Human-in-the-loop on file writes
- Config-driven tool selection: a LangSmith **assistant** can pin which search
  tool the graph uses. Module 5 (Engine) uses an assistant that selects the
  `easy_search` variant.

The graph is exported as a factory `agent(config)` so each assistant's
`config.configurable.search_tool` chooses its tools at run time. The default
("tavily") is the original behavior, so Modules 2-4 are unaffected.
"""

import json
import os
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from tavily import TavilyClient

from utils.models import model
from utils.search import resilient_tavily_search

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))


@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """Search the web for information on a given query.

    Args:
        query: Search query to execute.
    """
    # Resilient wrapper: retries on Tavily failure, then falls back to a
    # topic-matched canned response. See utils/search.py.
    return resilient_tavily_search(query, max_retries=2)


_easy_client: TavilyClient | None = None


@tool(parse_docstring=True)
def easy_search(query: str) -> str:
    """Search the web for information on a given query.

    Args:
        query: Search query to execute.
    """
    # Lightweight search: trim each hit down to the essentials before handing it
    # back to the model. We round-trip through JSON to normalize the payload.
    global _easy_client
    if _easy_client is None:
        _easy_client = TavilyClient()
    results = _easy_client.search(query, max_results=3)
    hits = results.get("results", [])
    payload = json.dumps([{"title": h["title"], "url": h["url"]} for h in hits])
    parsed = json.loads(payload)
    return "\n".join(f"- {h['title']} ({h['url']})" for h in parsed)


_SEARCH_TOOLS = {"tavily": tavily_search, "easy": easy_search}

_SUBAGENT_PROMPT = (
    f"You are a research assistant. Today is {datetime.now().strftime('%Y-%m-%d')}.\n"
    "Use tools to gather information. Structure findings with clear headings and "
    "inline citations.\nLimit to 3 search calls."
)


def backend_factory(rt):
    """FilesystemBackend for disk access, /memories/ routed to StoreBackend."""
    return CompositeBackend(
        default=FilesystemBackend(root_dir=AGENT_DIR, virtual_mode=True),
        routes={"/memories/": StoreBackend()},
    )


def agent(config: RunnableConfig | None = None):
    """Graph factory.

    `config.configurable` fields (pinned by assistants, so one deployed graph
    serves multiple configurations):
    - `search_tool`: "tavily" (default) or "easy".
    - `interrupts`: human-in-the-loop on file writes (default True). The Module 5
      `engine-demo` assistant sets this False so unattended seed runs don't pause.
    """
    configurable = (config or {}).get("configurable", {})
    web_search = _SEARCH_TOOLS.get(configurable.get("search_tool", "tavily"), tavily_search)
    interrupts = configurable.get("interrupts", True)

    research_subagent = {
        "name": "research-agent",
        "description": "Delegate research tasks. Give one topic at a time.",
        "system_prompt": _SUBAGENT_PROMPT,
        "tools": [web_search],
    }

    return create_deep_agent(
        model=model,
        tools=[web_search],
        system_prompt="You are an expert research assistant.",
        memory=["./AGENTS.md"],
        skills=["./skills/"],
        subagents=[research_subagent],
        backend=backend_factory,
        interrupt_on={"write_file": True, "edit_file": True} if interrupts else {},
    )
