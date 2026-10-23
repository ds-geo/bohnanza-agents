"""Structured JSON event logging for game activity."""

from __future__ import annotations

import json
import logging
from typing import Any


_EVENT_LOGGER = logging.getLogger("bohnanza.events")


def configure_json_logging(log_file: str | None = None) -> None:
    """Send structured game events to stderr or to ``log_file`` as JSON lines.

    Call this once from an application or game script. Keeping configuration
    opt-in avoids changing logging behavior for library users.
    """

    handler: logging.Handler
    if log_file is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    _EVENT_LOGGER.handlers.clear()
    _EVENT_LOGGER.addHandler(handler)
    _EVENT_LOGGER.setLevel(logging.INFO)
    _EVENT_LOGGER.propagate = False


def log_event(event: str, **data: Any) -> None:
    """Emit one JSON-encoded event through the standard logging system."""

    payload = {"event": event, **data}
    _EVENT_LOGGER.info(json.dumps(payload, sort_keys=True))
