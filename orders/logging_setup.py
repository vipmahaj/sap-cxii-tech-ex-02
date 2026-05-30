"""Structured JSON logging via structlog.

One JSON line per request. See ai-layer.md §1.6 for the schema we emit
from the /orders/ask endpoint.
"""

from __future__ import annotations

import logging

import structlog


def configure(level: str = "INFO") -> None:
    """Wire stdlib logging into structlog with JSON output."""
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
