"""Utilities for running text searches via DuckDuckGo."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Sequence

import requests
from ddgs import DDGS

LOGGER = logging.getLogger(__name__)
_DEFAULT_MAX_RESULTS = 5
_DEFAULT_RETRIES = 2
_DEFAULT_RETRY_DELAY = 1.0


def _search_once(query: str) -> List[Dict[str, str]]:
    """Execute a single DDGS text search and normalize its output."""
    with DDGS() as ddgs:
        raw_results = list(ddgs.text(query, max_results=_DEFAULT_MAX_RESULTS))
    return [
        {
            "title": result.get("title"),
            "href": result.get("href"),
            "body": result.get("body"),
        }
        for result in raw_results
    ]


def _search_with_retry(query: str) -> List[Dict[str, str]]:
    """Retry a search a few times before giving up."""
    last_exception: Exception | None = None
    for attempt in range(1, _DEFAULT_RETRIES + 2):
        try:
            return _search_once(query)
        except Exception as exc:  # pragma: no cover - non-deterministic network errors
            last_exception = exc
            LOGGER.warning(
                "Search for '%s' failed on attempt %s/%s: %s",
                query,
                attempt,
                _DEFAULT_RETRIES + 1,
                exc,
            )
            if attempt == _DEFAULT_RETRIES + 1:
                break
            time.sleep(_DEFAULT_RETRY_DELAY * attempt)
    LOGGER.error("Giving up on query '%s' due to: %s", query, last_exception)
    return []


async def start_text_search(
    queries: Sequence[str],
    max_workers: int = 5,
) -> List[List[Dict[str, str]]]:
    """
    Run several DuckDuckGo text searches in parallel.

    Args:
        queries: List of queries to execute.
        max_workers: Max number of parallel searches (default: 5).
    Returns:
        A list whose entries correspond to each query. Each entry is a list
        containing dicts with ``title``, ``href`` and ``body`` keys.
    """

    if not queries:
        return []

    loop = asyncio.get_running_loop()
    max_workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, partial(_search_with_retry, query))
            for query in queries
        ]
        return await asyncio.gather(*tasks)


def text_search(query: str) -> List[Dict[str, str]]:
    """Convenience wrapper to run a single search synchronously."""
    return _search_with_retry(query)


def fetch_html(url: str, timeout: float = 10.0) -> str:
    """
    Retrieve the HTML content for a given URL using ``requests``.

    Args:
        url: The URL to download.
        timeout: Optional timeout in seconds for the HTTP request.

    Returns:
        The response body decoded as text.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text
