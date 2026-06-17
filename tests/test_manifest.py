from __future__ import annotations

from importlib.resources import as_file

from uri_control import CapabilityRegistry

import urillm


def test_manifest_loads():
    with as_file(urillm.manifest_path()) as path:
        registry = CapabilityRegistry.from_manifest_files([path])
    assert registry.manifests[0].scheme == "llm"
    assert len(registry.routes) == 5
    ops = {route.operation for route in registry.routes}
    assert ops == {
        "llm.vision.analyze",
        "llm.text.plan",
        "llm.text.decide",
        "llm.chat.completion",
    }
