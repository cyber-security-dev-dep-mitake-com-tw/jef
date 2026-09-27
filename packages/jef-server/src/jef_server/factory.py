"""Module-level ASGI app for uvicorn's import string."""

from __future__ import annotations

from .app import create_app

app = create_app()
