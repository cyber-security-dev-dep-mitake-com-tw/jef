"""``python -m jef_server`` -- run the API with uvicorn."""

from __future__ import annotations

import logging
import os

import uvicorn


def main() -> None:
    logging.basicConfig(level=os.environ.get("JEF_LOG_LEVEL", "INFO"))
    uvicorn.run(
        "jef_server.factory:app",
        host=os.environ.get("JEF_HOST", "0.0.0.0"),  # noqa: S104 -- containerised by design
        port=int(os.environ.get("JEF_PORT", "8080")),
        workers=int(os.environ.get("JEF_WORKERS", "1")),
        log_level=os.environ.get("JEF_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
