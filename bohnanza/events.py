"""Structured JSON event logging for game activity."""

from __future__ import annotations

import json
import logging
from typing import Any


_EVENT_LOGGER = logging.getLogger("bohnanza.events")


def log_event(event: str, **data: Any) -> None:
    """Emit one JSON-encoded event through the standard logging system."""

    payload = {"event": event, **data}
    _EVENT_LOGGER.info(json.dumps(payload, sort_keys=True))