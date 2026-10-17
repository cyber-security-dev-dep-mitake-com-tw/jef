"""The server's version, read from the installed distribution.

In its own module so both the package and ``app.py`` can import it without
a cycle: ``__init__`` imports ``app``, and ``app`` must not import ``__init__``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("jef-server")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"
