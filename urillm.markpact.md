# UriPack: urillm

Self-contained Markpact — definitions, full source, run config. Unpack & run: `urisys markpact run urillm/urillm.markpact.md --as service` (writes `.markpact/`).

```yaml markpact:pack
apiVersion: urisys.io/v1
kind: UriPack
metadata:
  id: urillm-pack
  version: 1.0.0
  language: python
description: LLM vision, planning and chat for urisys-node (mock/heuristic/litellm/openai).
schemes:
- llm
capabilities:
- id: llm.vision.analyze
  uri: llm://{host}/vision/query/analyze
  kind: query
  operation: llm.vision.analyze
  handler: python://urillm.handlers:vision_analyze
  side_effects: false
  approval: not_required
- id: llm.text.plan
  uri: llm://{host}/text/query/plan
  kind: query
  operation: llm.text.plan
  handler: python://urillm.handlers:text_plan
  side_effects: false
  approval: not_required
- id: llm.text.decide
  uri: llm://{host}/text/query/decide
  kind: query
  operation: llm.text.decide
  handler: python://urillm.handlers:text_decide
  side_effects: false
  approval: not_required
- id: llm.chat.completion
  uri: llm://{host}/chat/query/completion
  kind: query
  operation: llm.chat.completion
  handler: python://urillm.handlers:chat_completion
  side_effects: false
  approval: not_required
- id: llm.chat.completion
  uri: llm://{host}/chat/command/completion
  kind: command
  operation: llm.chat.completion
  handler: python://urillm.handlers:chat_completion
  side_effects: true
  approval: required
policy:
  default: deny_mutations_without_approval
runtime:
  default_environment: mock
  supports:
  - mock
  - local
  - docker
```

```yaml markpact:run
modes:
- pack
- service
- flow
- interface
- adapter
default: service
scheme: llm
service:
  port: 8790
  wire: POST /uri/call
flow:
  ids:
  - llm-guided-gui-click
adapter:
  wire: POST /uri/call
  events: GET /events
```

```python markpact:module path=urillm/__init__.py
from __future__ import annotations

from importlib.resources import files

from .routes import register

__all__ = ["register", "manifest_path"]


def manifest_path():
    return files(__package__).joinpath("manifest.yaml")
```

```python markpact:module path=urillm/handlers.py
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from uri_control.edge.env import is_secret_env, resolve_env_var
from urioperators import (
    decision_from_parsed,
    litellm_chat,
    openai_compatible_chat,
    parse_json_response,
    plan_from_parsed,
)


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
        parsed = openai_compatible_chat(messages, model, api_key, base_url or 'https://api.openai.com/v1', temperature=0, max_tokens=1024, timeout=60)
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
        parsed = litellm_chat(messages, model, temperature=0, max_tokens=1024)
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


_OFFICE_PHRASE_MAP: list[tuple[str, str, dict]] = [
    ('kliknij ok', 'kvm://local/task/command/click-text', {'text': 'OK'}),
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


def _completion_litellm(messages, model, temperature, max_tokens):
    import litellm  # type: ignore
    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _completion_openai(messages, model, api_key, base_url, temperature, max_tokens, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def chat_completion(payload, context):
    """Generic LLM chat completion via llm://{host}/chat/query/completion."""
    messages = payload.get("messages")
    if not messages:
        return {"ok": False, "error": "payload.messages is required"}
    cfg = _llm_cfg(context)
    driver = payload.get("driver") or _driver(context) or "litellm"
    model = payload.get("model") or _env("model", cfg, context)
    temperature = float(payload.get("temperature") or _env("temperature", cfg, context) or "0.7")
    max_tokens = int(payload.get("max_tokens") or _env("max_tokens", cfg, context) or "1024")
    fmt = payload.get("format", "text")
    api_key = (
        _env("api_key", cfg, context)
        or resolve_env_var("OPENROUTER_API_KEY", context, secret=True)
        or resolve_env_var("OPENAI_API_KEY", context, secret=True)
    )
    base_url = _env("base_url", cfg, context)
    try:
        if driver == "litellm":
            if not model:
                return {"ok": False, "error": "model is required for litellm driver"}
            raw = _completion_litellm(messages, model, temperature, max_tokens)
        elif driver in ("openai", "openrouter"):
            if not api_key:
                return {"ok": False, "error": "api_key is required for openai driver"}
            if not base_url:
                base_url = (
                    "https://openrouter.ai/api/v1"
                    if resolve_env_var("OPENROUTER_API_KEY", context, secret=True)
                    else "https://api.openai.com/v1"
                )
            raw = _completion_openai(messages, model, api_key, base_url, temperature, max_tokens, 90.0)
        elif driver == "mock":
            raw = f"[mock] {messages[-1].get('content', '')[:100]}..."
        else:
            return {"ok": False, "error": f"unsupported driver: {driver}"}
        content = parse_json_response(raw) if fmt == "json" else raw
        return {
            "ok": True,
            "content": content,
            "format": fmt,
            "model": model,
            "driver": driver,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "driver": driver}


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
                    parsed = litellm_chat(messages, model, temperature=temperature, max_tokens=max_tokens)
                else:
                    if not base_url:
                        base_url = (
                            'https://openrouter.ai/api/v1'
                            if resolve_env_var('OPENROUTER_API_KEY', context, secret=True)
                            else 'https://api.openai.com/v1'
                        )
                    parsed = openai_compatible_chat(
                        messages, model, api_key, base_url, temperature=temperature, max_tokens=max_tokens,
                    )
                planned = plan_from_parsed(parsed, model, transcript)
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


def _decide_messages(question: str, context_value) -> list[dict[str, str]]:
    context_text = json.dumps(context_value, ensure_ascii=False, default=str)
    if len(context_text) > 12000:
        context_text = context_text[:12000] + '…'
    prompt = (
        'You are a runtime judge for automation workflows. '
        'Return JSON only with keys: ok (bool), decision (retry|abort), reason (string), confidence (0-1). '
        f'Question: {question}\n'
        f'Context:\n{context_text}'
    )
    return [{'role': 'user', 'content': prompt}]


def _mock_decide(question: str, context_value) -> dict[str, Any]:
    blob = json.dumps(context_value or {}, ensure_ascii=False, default=str).lower()
    retry = 'error' in blob or '502' in blob
    return {
        'ok': retry,
        'decision': 'retry' if retry else 'abort',
        'reason': 'mock-decide: critical pattern in context' if retry else 'mock-decide: no critical pattern',
        'confidence': 0.8 if retry else 0.9,
        'model': 'mock-decide',
        'question': question,
    }


def text_decide(payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """NL judge for log/workflow context — llm://{host}/text/query/decide."""
    cfg = _llm_cfg(context)
    driver = payload.get('driver') or _driver(context) or 'mock'
    question = str(payload.get('question') or '').strip()
    context_value = payload.get('context')

    if not question:
        return {'ok': False, 'error': 'payload.question is required'}

    if context.get('dry_run') or not _real_allowed(context) or driver in ('mock', 'heuristic'):
        return _mock_decide(question, context_value)

    model = _env('model', cfg, context)
    api_key = (
        _env('api_key', cfg, context)
        or resolve_env_var('OPENROUTER_API_KEY', context, secret=True)
        or resolve_env_var('OPENAI_API_KEY', context, secret=True)
    )
    base_url = _env('base_url', cfg, context)
    temperature = float(_env('temperature', cfg, context) or '0')
    max_tokens = int(_env('max_tokens', cfg, context) or '512')
    messages = _decide_messages(question, context_value)

    if not model or not api_key:
        return _mock_decide(question, context_value)

    try:
        if driver == 'litellm':
            if not str(model).startswith('openrouter/') and resolve_env_var('OPENROUTER_API_KEY', context, secret=True):
                model = f'openrouter/{model.lstrip("openrouter/")}'
            parsed = litellm_chat(messages, model, temperature=temperature, max_tokens=max_tokens)
        elif driver in ('openai', 'openrouter'):
            if not base_url:
                base_url = (
                    'https://openrouter.ai/api/v1'
                    if resolve_env_var('OPENROUTER_API_KEY', context, secret=True)
                    else 'https://api.openai.com/v1'
                )
            parsed = openai_compatible_chat(
                messages, model, api_key, base_url, temperature=temperature, max_tokens=max_tokens,
            )
        else:
            return _mock_decide(question, context_value)
        return decision_from_parsed(parsed, model, question)
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError, RuntimeError):
        return _mock_decide(question, context_value)
```

```python markpact:module path=urillm/routes.py
from __future__ import annotations

from importlib.resources import files

from uri_control.edge.manifest import register_manifest_file


def register(runtime):
    register_manifest_file(runtime, files(__package__).joinpath("manifest.yaml"))
```

```yaml markpact:flow id=llm-guided-gui-click
flow:
  id: llm-guided-gui-click
  description: Screenshot, OCR, LLM vision analyze, then click Install (KVM + OCR + LLM).

defaults:
  approved: true
  dry_run: true

do:
  - kvm://local/monitor/primary/query/screenshot
  - ocr://local/image/latest/query/text
  - llm://local/vision/query/analyze:
      target_text: Install
  - kvm://local/task/command/click-text:
      text: Install
```

```markdown markpact:docs
# urillm
```

