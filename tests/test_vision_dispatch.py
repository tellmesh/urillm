"""Characterization tests for the LLM-vision driver dispatch (no tesseract/network).

These pin the decision behaviour of urillm.handlers.vision_analyze across every
driver branch, so the dispatch refactor (driver table + per-driver functions)
cannot silently change what gets clicked. They feed OCR boxes directly instead
of running tesseract, so they are CI-safe.
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "packages" / "python"
CANONICAL = Path(__file__).resolve().parents[2] / "packages" / "python"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(CANONICAL))

import urillm.handlers as h  # noqa: E402


def _ctx(driver: str) -> dict:
    return {"config": {"llm": {"driver": driver}}}


def test_mock_matches_target_box():
    out = h.vision_analyze(
        {"goal": "click OK", "target_text": "OK",
         "ocr": {"boxes": [{"text": "Cancel", "x": 10, "y": 10, "w": 40, "h": 20},
                           {"text": "OK", "x": 100, "y": 50, "w": 30, "h": 20}]}},
        _ctx("mock"),
    )
    assert out == {"action": "click", "target_text": "OK", "x": 115, "y": 60,
                   "confidence": 0.9, "source": "mock-llm"}


def test_heuristic_goal_substring_match():
    out = h.vision_analyze(
        {"goal": "click Submit",
         "ocr": {"boxes": [{"text": "submit now", "x": 5, "y": 5, "w": 60, "h": 20}]}},
        _ctx("heuristic"),
    )
    assert out["action"] == "click"
    assert out["target_text"] == "submit now"
    assert out["source"] == "heuristic"


def test_no_target_match_falls_back_to_first_box():
    out = h.vision_analyze(
        {"goal": "click Missing", "target_text": "Missing",
         "ocr": {"boxes": [{"text": "Foo", "x": 0, "y": 0, "w": 10, "h": 10}]}},
        _ctx("mock"),
    )
    assert out["action"] == "click"
    assert out["confidence"] == 0.35
    assert out["source"] == "mock-llm-fallback"


def test_empty_boxes_yields_no_action():
    out = h.vision_analyze(
        {"goal": "click X", "target_text": "X", "ocr": {"boxes": []}},
        _ctx("heuristic"),
    )
    assert out == {"action": "none", "confidence": 0.0, "source": "heuristic"}


def test_unknown_driver_uses_heuristic():
    out = h.vision_analyze(
        {"goal": "click OK", "target_text": "OK",
         "ocr": {"boxes": [{"text": "OK", "x": 2, "y": 2, "w": 8, "h": 8}]}},
        _ctx("totally-unknown"),
    )
    assert out["action"] == "click"
    assert out["source"] == "heuristic"


def test_openai_without_key_falls_back_to_heuristic(monkeypatch):
    # resolve_env_var falls back to os.environ, so isolate the keys for determinism.
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    out = h.vision_analyze({"goal": "click OK", "target_text": "OK"}, _ctx("openai"))
    assert out == {"action": "none", "confidence": 0.0, "source": "heuristic-fallback"}
