from urillm.handlers import text_plan


def test_plan_scroll_phrase_map():
    result = text_plan(
        {'transcript': 'scroll down slightly', 'allowed_schemes': ['him']},
        {'approved': True, 'dry_run': False, 'allow_real': True},
    )
    assert result['ok']
    assert result['uri'] == 'him://local/mouse/command/scroll'
    assert result['payload']['amount'] == -3
    assert result['model'] == 'phrase-map'


def test_plan_type_letter():
    result = text_plan(
        {'transcript': "type exactly one letter 'z'", 'allowed_schemes': ['him']},
        {'approved': True, 'dry_run': False, 'allow_real': True},
    )
    assert result['ok']
    assert result['uri'] == 'him://local/keyboard/command/type'
    assert result['payload']['text'] == 'z'


def test_plan_rejects_disallowed_scheme():
    result = text_plan(
        {'transcript': 'move mouse', 'allowed_schemes': ['kvm']},
        {'approved': True, 'dry_run': False, 'allow_real': True},
    )
    assert not result['ok']
    assert 'him' in result['error']
