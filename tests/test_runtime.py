from __future__ import annotations

from uri_control.edge.runtime import Runtime

import urillm


def test_text_plan_phrase_map():
    rt = Runtime(config={"llm": {"driver": "mock"}})
    urillm.register(rt)
    res = rt.call(
        "llm://local/text/query/plan",
        {"transcript": "scroll down", "allowed_schemes": ["him"]},
        {"params": {"host": "local"}},
    )
    assert res["ok"]
    assert res["result"]["uri"] == "him://local/mouse/command/scroll"


def test_text_decide_mock():
    rt = Runtime(config={"llm": {"driver": "mock"}})
    urillm.register(rt)
    res = rt.call(
        "llm://local/text/query/decide",
        {"question": "retry?", "context": {"error": "502"}},
        {"params": {"host": "local"}},
    )
    assert res["ok"]
    assert res["result"]["decision"] == "retry"
