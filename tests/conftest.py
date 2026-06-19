"""Ensure uri_resolver + uri_control are importable when testing from urisys venv."""

from __future__ import annotations

import sys
from pathlib import Path


def _tellmesh_root() -> Path | None:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "uricontrol").is_dir() and (candidate / "uriresolver").is_dir():
            return candidate
    return None


def _ensure_siblings() -> None:
    root = _tellmesh_root()
    if root is None:
        return
    for rel in ("uriresolver/src", "uricontrol/core/python"):
        path = str((root / rel).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


_ensure_siblings()
