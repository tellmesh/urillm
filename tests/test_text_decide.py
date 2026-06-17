
def test_text_decide_mock_retry_on_error_context():
    from urillm.handlers import text_decide

    out = text_decide(
        {"question": "Should we retry?", "context": {"status": 502, "error": "bad gateway"}},
        {"approved": True, "dry_run": False, "allow_real": False},
    )
    assert out["ok"] is True
    assert out["decision"] == "retry"


def test_text_decide_requires_question():
    from urillm.handlers import text_decide

    out = text_decide({}, {})
    assert out["ok"] is False
