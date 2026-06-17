from __future__ import annotations

from importlib.resources import files

from .routes import register

__all__ = ["register", "manifest_path"]


def manifest_path():
    return files(__package__).joinpath("manifest.yaml")
