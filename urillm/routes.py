from __future__ import annotations

from importlib.resources import files

from urisysedge.manifest import register_manifest_file


def register(runtime):
    register_manifest_file(runtime, files(__package__).joinpath("manifest.yaml"))
