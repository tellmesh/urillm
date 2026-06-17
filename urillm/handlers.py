from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from urisysedge.env import is_secret_env, resolve_env_var


def _llm_cfg(context):
    return context.get('config', {}).get('llm', {})


def _driver(context):
    return _llm_cfg(context).get('driver', 'mock')


def _goal_text(payload):
    return (payload.get('goal') or payload.get('instruction') or payload.get('target_text') or '').strip()


def _target_from_goal(goal: str) -> str:
    goal = goal.lower().strip()
    for prefix in ('click ', 'tap ', 'press ', 'select ', 'find '):
        if goal.startswith(prefix):
            return goal[len(prefix):].strip().strip('"\'')
    return goal


def _box_center(box):
    return int(box['x'] + box.get('w', 0) / 2), int(box['y'] + box.get('h', 0) / 2)


def _click_box(box, confidence, source):
    x, y = _box_center(box)
    return {
        'action': 'click',
        'target_text': box.get('text'),
        'x': x,
        'y': y,
        'confidence': confidence,
        'source': source,
    }


def _find_target_box(boxes, target):
    for box in boxes:
        text = (box.get('text') or '').lower()
        if text == target or target in text or text in target:
            return box
    return None


def _find_goal_box(boxes, goal):
    for box in boxes:
        text = (box.get('text') or '').lower()
        if text and (text in goal or goal in text):
            return box
    return None


def _heuristic_analyze(payload, source='heuristic'):
    goal = _goal_text(payload).lower()
    target = (payload.get('target_text') or _target_from_goal(goal)).lower()
    boxes = payload.get('ocr', {}).get('boxes') or payload.get('boxes') or []
    box = _find_target_box(boxes, target) if target else None
    if box:
        return _click_box(box, float(box.get('confidence', 0.9)), source)
    box = _find_goal_box(boxes, goal)
    if box:
        return _click_box(box, float(box.get('confidence', 0.85)), source)
    if boxes:
        return _click_box(boxes[0], 0.35, f'{source}-fallback')
    return {'action': 'none', 'confidence': 0.0, 'source': source}


def _parse_json_response(text: str) -> dict:
    text = (text or '').strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _openai_chat(messages, model, api_key, base_url=None):
    url = (base_url or 'https://api.openai.com/v1').rstrip('/') + '/chat/completions'
    body = json.dumps({'model': model, 'messages': messages, 'temperature': 0}).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    content = data['choices'][0]['message']['content']
    return _parse_json_response(content)


def _litellm_chat(messages, model):
    try:
        import litellm  # type: ignore
    except Exception as exc:
        raise RuntimeError('litellm driver requires: pip install litellm') from exc
    response = litellm.completion(model=model, messages=messages, temperature=0)
    content = response.choices[0].message.content
    return _parse_json_response(content)


def _vision_messages(goal, target, shot, ocr):
    target = target or _target_from_goal(goal)
    prompt = (
        f'You are a UI automation assistant. Goal: {goal}. '
        f'Target text: {target or "unspecified"}. '
        'Return JSON only with keys action, x, y, target_text, confidence. '
        'Use action=click when a clickable target is found, otherwise action=none. '
        'Coordinates must be pixel center of the target in the screenshot.'
    )
    ocr_text = (ocr or {}).get('text') or ''
    ocr_boxes = (ocr or {}).get('boxes') or []
    if ocr_boxes:
        prompt += f' OCR text: {ocr_text}. OCR boxes: {json.dumps(ocr_boxes[:40])}.'
    content = [{'type': 'text', 'text': prompt}]
    mime = (shot or {}).get('mime')
    b64 = (shot or {}).get('base64')
    if mime == 'image/png' and b64:
        content.append({'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}})
    return [{'role': 'user', 'content': content}]


def _normalize_action(parsed, source):
    if not parsed:
        return {'action': 'none', 'confidence': 0.0, 'source': source}
    action = (parsed.get('action') or 'none').lower()
    if action != 'click':
        return {'action': 'none', 'confidence': float(parsed.get('confidence', 0.0)), 'source': source}
    return {
        'action': 'click',
        'target_text': parsed.get('target_text'),
        'x': int(parsed['x']),
        'y': int(parsed['y']),
        'confidence': float(parsed.get('confidence', 0.8)),
        'source': source,
    }


def _analyze_openai(payload, context, *, goal, target, shot, ocr, cfg):
    api_key = (
        resolve_env_var('OPENROUTER_API_KEY', context, secret=True)
        or resolve_env_var('OPENAI_API_KEY', context, secret=True)
        or cfg.get('api_key')
    )
    if not api_key:
        return _heuristic_analyze(payload, source='heuristic-fallback')
    model = cfg.get('model') or resolve_env_var('LLM_MODEL', context) or 'gpt-4o-mini'
    base_url = cfg.get('base_url') or resolve_env_var('LLM_BASE_URL', context)
    if not base_url and resolve_env_var('OPENROUTER_API_KEY', context, secret=True):
        base_url = 'https://openrouter.ai/api/v1'
    messages = _vision_messages(goal, target, shot, ocr)
    try:
        parsed = _openai_chat(messages, model, api_key, base_url=base_url)
        return _normalize_action(parsed, source=f'openai:{model}')
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError):
        return _heuristic_analyze(payload, source='heuristic-fallback')


def _analyze_litellm(payload, context, *, goal, target, shot, ocr, cfg):
    model = cfg.get('model') or resolve_env_var('LLM_MODEL', context)
    if not model:
        raise ValueError('llm.model is required when llm.driver=litellm')
    if not str(model).startswith('openrouter/') and resolve_env_var('OPENROUTER_API_KEY', context, secret=True):
        model = f'openrouter/{model.lstrip("openrouter/")}'
    messages = _vision_messages(goal, target, shot, ocr)
    try:
        parsed = _litellm_chat(messages, model)
        return _normalize_action(parsed, source=f'litellm:{model}')
    except Exception:
        return _heuristic_analyze(payload, source='heuristic-fallback')


# driver -> analyzer. mock/heuristic and any unknown driver use the OCR heuristic.
_VISION_DRIVERS = {
    'openai': _analyze_openai,
    'litellm': _analyze_litellm,
}


def _vision_analyze(payload, context):
    driver = _driver(context)
    if driver == 'mock':
        return _heuristic_analyze(payload, source='mock-llm')
    analyzer = _VISION_DRIVERS.get(driver)
    if analyzer is None:
        return _heuristic_analyze(payload, source='heuristic')
    cfg = _llm_cfg(context)
    goal = _goal_text(payload)
    return analyzer(
        payload,
        context,
        goal=goal,
        target=payload.get('target_text') or _target_from_goal(goal),
        shot=context.get('state', {}).get('latest_screenshot') or {},
        ocr=payload.get('ocr') or {},
        cfg=cfg,
    )


def vision_analyze(payload, context):
    return _vision_analyze(payload, context)


def _real_allowed(context):
    return bool(context.get('allow_real') or os.environ.get('URISYS_ALLOW_REAL') == '1')


def _env(name, cfg, context, default=None):
    env_name = cfg.get(f'{name}_env') or name.upper()
    explicit = cfg.get(name)
    if explicit is not None:
        return str(explicit)
    return resolve_env_var(env_name, context, secret=is_secret_env(env_name), default=default)


def _openai_compatible_chat(messages, model, api_key, base_url, temperature=0.0, max_tokens=512):
    url = base_url.rstrip('/') + '/chat/completions'
    body = json.dumps({
        'model': model,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return _parse_json_response(data['choices'][0]['message']['content'])


def _litellm_plan_chat(messages, model, temperature=0.0, max_tokens=512):
    try:
        import litellm  # type: ignore
    except Exception as exc:
        raise RuntimeError('litellm driver requires: pip install litellm') from exc
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _parse_json_response(response.choices[0].message.content)


_OFFICE_PHRASE_MAP: list[tuple[str, str, dict]] = [
    ('scroll down', 'him://local/mouse/command/scroll', {'amount': -3}),
    ('scroll up', 'him://local/mouse/command/scroll', {'amount': 3}),
    ('scroll', 'him://local/mouse/command/scroll', {'amount': -3}),
    ('przewiń', 'him://local/mouse/command/scroll', {'amount': -3}),
    ('move mouse', 'him://local/mouse/command/move', {'x': 400, 'y': 300}),
    ('rusz mysz', 'him://local/mouse/command/move', {'x': 400, 'y': 300}),
]


def _match_office_transcript(text: str, allowed: list[str] | None = None) -> tuple[str, dict]:
    lowered = (text or '').lower().strip()
    schemes = {s.lower() for s in allowed} if allowed else None

    if not schemes or "urimail" in schemes:
        if any(k in lowered for k in ("unread", "summarize", "summary", "mail", "inbox", "draft", "reply")):
            return "urimail://local/message/command/compose", {
                "subject": "Re: Weekly report",
                "body": "Thanks — I'll review and reply tomorrow.",
            }

    if not schemes or "urioffice" in schemes:
        if any(k in lowered for k in ("writer", "document", "paragraph", "docx", "quarterly", "report")):
            return "urioffice://local/writer/command/render", {
                "title": "writer-output",
                "text": "Quarterly results show steady growth across all regions.",
                "format": "txt",
            }

    for phrase, uri, payload in _OFFICE_PHRASE_MAP:
        if phrase in lowered:
            out = dict(payload)
            coords = re.search(r'x[:\s=]+(\d+)[^\d]+y[:\s=]+(\d+)', lowered)
            if coords and 'move' in uri:
                out.update({'x': int(coords.group(1)), 'y': int(coords.group(2))})
            letter = re.search(r"letter ['\"]?([a-z])['\"]?", lowered)
            if letter and 'type' in lowered:
                return 'him://local/keyboard/command/type', {'text': letter.group(1)}
            scroll_amount = re.search(r'amount[:\s=]+(-?\d+)', lowered)
            if scroll_amount and 'scroll' in uri:
                out['amount'] = int(scroll_amount.group(1))
            return uri, out
    letter = re.search(r"letter ['\"]?([a-z])['\"]?", lowered)
    if letter or re.search(r'\btype\b', lowered):
        ch = letter.group(1) if letter else 'a'
        return 'him://local/keyboard/command/type', {'text': ch}
    if 'move' in lowered or 'mysz' in lowered:
        coords = re.search(r'x[:\s=]+(\d+)[^\d]+y[:\s=]+(\d+)', lowered)
        if coords:
            return 'him://local/mouse/command/move', {'x': int(coords.group(1)), 'y': int(coords.group(2))}
        return 'him://local/mouse/command/move', {'x': 400, 'y': 300}
    if 'scroll' in lowered or 'przewi' in lowered:
        amount = -3 if 'down' in lowered or 'w dół' in lowered or 'dół' in lowered else 3
        return 'him://local/mouse/command/scroll', {'amount': amount}
    return 'him://local/keyboard/command/type', {'text': ' '}


def _plan_messages(transcript: str, allowed: list[str] | None) -> list[dict]:
    schemes = ', '.join(allowed) if allowed else 'him, kvm, browser, screen'
    examples = '\n'.join(
        f'- "{phrase}" -> {uri} payload={json.dumps(payload)}'
        for phrase, uri, payload in _OFFICE_PHRASE_MAP
    )
    prompt = (
        'You map text commands to urisys URI calls for safe desktop office simulation. '
        'Prefer small mouse moves, single-letter typing, and gentle scroll. '
        'Never plan clicks on buttons, Enter, Alt+F4, or destructive actions. '
        'Return JSON only with keys: uri (string), payload (object). '
        f'Allowed URI schemes (segment before ://): {schemes}.\n'
        f'Examples:\n{examples}\n'
        f'Transcript: {transcript}'
    )
    return [{'role': 'user', 'content': prompt}]


def _plan_from_parsed(parsed: dict, model: str, transcript: str) -> dict:
    uri = str(parsed.get('uri') or '').strip()
    inner_payload = parsed.get('payload') if isinstance(parsed.get('payload'), dict) else {}
    if not uri:
        raise ValueError('missing uri in LLM plan response')
    return {
        'ok': True,
        'uri': uri,
        'payload': inner_payload,
        'transcript': transcript,
        'model': model,
    }


def text_plan(payload, context):
    transcript = str(payload.get('transcript') or payload.get('text') or '').strip()
    if not transcript:
        return {'ok': False, 'error': 'payload.transcript is required'}
    allowed = payload.get('allowed_schemes')
    schemes = [str(s).strip() for s in allowed] if isinstance(allowed, list) else None

    cfg = _llm_cfg(context)
    driver = _driver(context)
    model_used = 'phrase-map'
    uri, inner_payload = _match_office_transcript(transcript, schemes)

    if not context.get('dry_run') and _real_allowed(context) and driver not in ('mock', 'heuristic'):
        model = _env('model', cfg, context)
        api_key = (
            _env('api_key', cfg, context)
            or resolve_env_var('OPENROUTER_API_KEY', context, secret=True)
            or resolve_env_var('OPENAI_API_KEY', context, secret=True)
        )
        base_url = _env('base_url', cfg, context)
        temperature = float(_env('temperature', cfg, context) or '0')
        max_tokens = int(_env('max_tokens', cfg, context) or '512')
        if model and api_key and driver in ('litellm', 'openai', 'openrouter'):
            messages = _plan_messages(transcript, schemes)
            try:
                if driver == 'litellm':
                    if not str(model).startswith('openrouter/') and resolve_env_var('OPENROUTER_API_KEY', context, secret=True):
                        model = f'openrouter/{model.lstrip("openrouter/")}'
                    parsed = _litellm_plan_chat(messages, model, temperature=temperature, max_tokens=max_tokens)
                else:
                    if not base_url:
                        base_url = (
                            'https://openrouter.ai/api/v1'
                            if resolve_env_var('OPENROUTER_API_KEY', context, secret=True)
                            else 'https://api.openai.com/v1'
                        )
                    parsed = _openai_compatible_chat(
                        messages, model, api_key, base_url, temperature, max_tokens,
                    )
                planned = _plan_from_parsed(parsed, model, transcript)
                uri = planned['uri']
                inner_payload = planned['payload']
                model_used = str(model)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError, RuntimeError):
                pass

    scheme = uri.split('://', 1)[0]
    if schemes and scheme not in schemes:
        return {
            'ok': False,
            'error': f'scheme {scheme!r} not in allowed_schemes',
            'uri': uri,
            'payload': inner_payload,
            'transcript': transcript,
        }
    return {
        'ok': True,
        'uri': uri,
        'payload': inner_payload,
        'transcript': transcript,
        'model': model_used,
    }
