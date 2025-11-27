"""LangChain-based data generation agent."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
import logging
from typing import Any, Callable, Dict, List

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from .scrap import fetch_html, text_search

LOGGER = logging.getLogger(__name__)
DEFAULT_SYSTEM_PROMPT = (
    "You are a research assistant that MUST use web search and URL fetching tools "
    "before generating each dataset entry. Always search first, then fetch one of "
    "the discovered URLs to ground your writing."
)


SearchFn = Callable[[str], Sequence[Dict[str, str]]]
FetchFn = Callable[[str], str]


class DataGenAgent:
    """Generate synthetic data items based on retrieval-augmented seeds."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        search_fn: SearchFn = text_search,
        fetch_fn: FetchFn = fetch_html,
        search_result_limit: int = 3,
        max_fetch_chars: int = 4000,
        system_prompt: str | None = None,
    ) -> None:
        if model is None:
            raise ValueError("A LangChain chat model must be provided.")
        if search_result_limit <= 0:
            raise ValueError("search_result_limit must be positive.")
        if max_fetch_chars <= 0:
            raise ValueError("max_fetch_chars must be positive.")

        self._model = model
        self._search_fn = search_fn
        self._fetch_fn = fetch_fn
        self._search_result_limit = search_result_limit
        self._max_fetch_chars = max_fetch_chars
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._agent = create_agent(
            model=self._model,
            tools=self._build_tools(),
            system_prompt=self._system_prompt,
        )

    def generate(self, prompt: str, count: int) -> List[str]:
        """
        Generate ``count`` dataset items following ``prompt`` instructions.

        Args:
            prompt: High-level generation goal the agent should follow.
            count: Number of items to create.
        Returns:
            List of generated strings, one per requested item.
        """

        if count <= 0:
            return []

        items: List[str] = []
        for idx in range(count):
            user_message = self._build_user_prompt(prompt, idx + 1, count)
            LOGGER.debug("Invoking DataGenAgent for item %s/%s", idx + 1, count)
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]}
            )
            items.append(self._extract_text(result))
        return items

    def _build_tools(self) -> List:
        """Create LangChain tool objects for search + fetch."""
        search_fn = self._search_fn
        fetch_fn = self._fetch_fn
        result_limit = self._search_result_limit
        max_chars = self._max_fetch_chars

        @tool("seed_search")
        def seed_search(query: str) -> str:
            """Search DuckDuckGo and return top seed candidates as JSON."""
            records = search_fn(query)[:result_limit]
            if not records:
                return "[]"
            normalized = []
            for record in records:
                normalized.append(
                    {
                        "title": record.get("title"),
                        "href": record.get("href"),
                        "body": (record.get("body") or "")[:500],
                    }
                )
            return json.dumps(normalized, ensure_ascii=False)

        @tool("fetch_url")
        def fetch_url(url: str) -> str:
            """Download the HTML body for ``url``."""
            html = fetch_fn(url)
            if len(html) > max_chars:
                return html[:max_chars]
            return html

        return [seed_search, fetch_url]

    @staticmethod
    def _build_user_prompt(prompt: str, item_index: int, total_items: int) -> str:
        return (
            f"Goal: {prompt}\n"
            f"You must produce dataset item {item_index} of {total_items}.\n"
            "For every item:\n"
            "1. Think of a new short search query describing a niche angle of the goal.\n"
            "2. Call `seed_search` with that query.\n"
            "3. Select a promising URL from the results and call `fetch_url` to read it.\n"
            "4. Use that fetched content as the seed and write one high-quality data item.\n"
            "Return only the generated item text."
        )

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Extract the latest AI message text from the agent response."""
        if isinstance(result, dict):
            messages = result.get("messages")
            text = DataGenAgent._text_from_messages(messages)
            if text:
                return text
            if "output" in result and isinstance(result["output"], str):
                return result["output"]
        elif isinstance(result, (AIMessage, BaseMessage)):
            return DataGenAgent._message_content_to_text(result)
        elif isinstance(result, Sequence):
            text = DataGenAgent._text_from_messages(result)
            if text:
                return text
        raise ValueError("Unable to extract agent output text from response.")

    @staticmethod
    def _text_from_messages(messages: Iterable[BaseMessage | Any] | None) -> str:
        if not messages:
            return ""
        last_ai: str = ""
        for message in messages:
            if isinstance(message, AIMessage):
                last_ai = DataGenAgent._message_content_to_text(message)
        return last_ai

    @staticmethod
    def _message_content_to_text(message: BaseMessage | AIMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            fragments = []
            for chunk in content:
                if isinstance(chunk, dict):
                    text = chunk.get("text")
                    if text:
                        fragments.append(text)
            return "\n".join(fragments).strip()
        return str(content).strip()
