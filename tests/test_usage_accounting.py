import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llmx.cli_backends import (
    CliBackendFailure,
    _codex_session_id,
    _latest_codex_rollout_usage,
    _parse_claude_json,
    cli_chat,
)
from llmx.providers import LlmxError, _normalize_usage
from llmx.usage_report import output_tokens

FIXTURES = Path(__file__).parent / "fixtures"


def _native_sse_response(message):
    import io
    events = [{"type": "message_start", "message": {"model": message["model"], "usage": message.get("usage", {})}}]
    events.extend({"type": "content_block_delta", "delta": {"type": "text_delta", "text": block["text"]}}
                  for block in message.get("content", []) if block["type"] == "text")
    events.extend([{"type": "message_delta", "delta": {"stop_reason": message["stop_reason"]}}, {"type": "message_stop"}])
    return io.BytesIO("".join("data: " + json.dumps(event) + "\n\n" for event in events).encode())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("termination", ["end_turn", "max_tokens", "incomplete", "reset"])
def test_native_wire_sse_for_both_modes_retains_safe_diagnostics(tmp_path, capsys, stream, termination):
    import io
    from llmx.providers import _openai_chat, ModelError, ServiceUnavailableError
    events = [{"type": "message_start", "message": {"model": "served-opus", "usage": {"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}},
              {"type": "ping"},
              {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "private-thinking-sentinel"}},
              {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "provisional"}}]
    if termination not in {"incomplete", "reset"}:
        events.extend([{"type": "message_delta", "delta": {"stop_reason": termination}, "usage": {"output_tokens": 12}}, {"type": "message_stop"}])
    class Response(io.BytesIO):
        headers = {"request-id": "req_test-123", "authorization": "private-header-sentinel"}
        def __iter__(self):
            for event in events:
                yield ("data: " + json.dumps(event) + "\n\n").encode()
            if termination == "reset":
                raise ConnectionResetError(54, "private-exception-sentinel")
    # Real HTTP iteration yields separate lines, including the blank separator.
    class Lines(Response):
        def __iter__(self):
            for payload in super().__iter__():
                yield from payload.splitlines(keepends=True)
    def respond(request, **kwargs):
        assert json.loads(request.data)["stream"] is True
        return Lines()
    ledger = tmp_path / "usage.jsonl"
    with patch("urllib.request.urlopen", side_effect=respond), patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.usage_log._LOG_PATH", ledger):
        if termination == "end_turn":
            assert _openai_chat("hi", "claude-opus-5-5", "anthropic-direct", None, .7, 30, stream, None, None, "max") == "provisional"
        else:
            with pytest.raises(ServiceUnavailableError if termination == "reset" else ModelError) as caught:
                _openai_chat("hi", "claude-opus-5-5", "anthropic-direct", None, .7, 30, stream, None, None, "max")
            assert caught.value.request_id == "req_test-123"
    captured = capsys.readouterr()
    if not stream and termination != "end_turn":
        assert captured.out == ""
    row = json.loads(ledger.read_text())
    assert row["request_id"] == "req_test-123" and row["served_model"] == "served-opus"
    assert row["prompt_tokens"] == 10
    assert bool(row["error"]) == (termination != "end_turn")
    if termination == "reset":
        assert row["transport_error_class"] == "ConnectionResetError"
        assert row["transport_errno"] == 54 and row["transport_phase"] == "read"
    assert "private-" not in ledger.read_text() + captured.out + captured.err


@pytest.mark.parametrize("stop,text", [("end_turn", "verdict"), ("max_tokens", "partial"), ("max_tokens", "")])
def test_native_claude_request_and_terminal_accounting(tmp_path, stop, text):
    from llmx.providers import _openai_chat, ModelError
    payload = {"model": "served-opus", "stop_reason": stop,
               "content": [{"type": "text", "text": text}],
               "usage": {"input_tokens": 10, "cache_read_input_tokens": 20,
                         "cache_creation_input_tokens": 30, "output_tokens": 40}}
    ledger = tmp_path / "usage.jsonl"
    def respond(request, **kwargs):
        body = json.loads(request.data)
        assert body["output_config"]["effort"] == "max"
        assert body["max_tokens"] == 128000  # max effort takes the whole output ceiling
        assert request.full_url == "https://api.anthropic.com/v1/messages"
        return _native_sse_response(payload)
    with patch("urllib.request.urlopen", side_effect=respond), patch(
        "llmx.providers._get_api_key", return_value="fake-key"
    ), patch("llmx.providers.OpenAI", side_effect=AssertionError("compat route forbidden")), patch(
        "llmx.usage_log._LOG_PATH", ledger
    ):
        if stop == "end_turn":
            assert _openai_chat("hi", "claude-opus-5-5", "anthropic-direct", None, .7,
                                30, False, None, None, "max") == text
        else:
            with pytest.raises(ModelError):
                _openai_chat("hi", "claude-opus-5-5", "anthropic-direct", None, .7,
                             30, False, None, None, "max")
    row = json.loads(ledger.read_text())
    assert row["served_model"] == "served-opus"
    assert row["stop_reason"] == stop
    assert row["prompt_tokens"] == 60
    assert row["cached_tokens"] == 20 and row["cache_write_tokens"] == 30
    assert bool(row["error"]) == (stop != "end_turn")


@pytest.mark.parametrize("stop,terminal", [("end_turn", True), ("max_tokens", True), ("refusal", True), ("end_turn", False)])
def test_native_claude_stream_preserves_partial_failure(tmp_path, stop, terminal):
    import io
    from llmx.api import LLM
    from llmx.providers import ModelError
    events = [{"type": "message_start", "message": {"model": "served-opus", "usage": {"input_tokens": 10}}},
              {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "partial"}},
              {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 12}}]
    if terminal:
        events.append({"type": "message_stop"})
    payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()
    ledger = tmp_path / "usage.jsonl"
    with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)), patch(
        "llmx.providers._get_api_key", return_value="fake"
    ), patch("llmx.api.check_api_key"), patch("llmx.spend_guard.enforce_daily_cap"), patch(
        "llmx.usage_log._LOG_PATH", ledger
    ):
        iterator = LLM(provider="anthropic-direct", model="claude-opus-5-5", auth="api").stream("hi")
        assert next(iterator) == "partial"
        if stop == "end_turn" and terminal:
            assert list(iterator) == []
        else:
            with pytest.raises(ModelError):
                list(iterator)
    row = json.loads(ledger.read_text())
    assert row["served_model"] == "served-opus" and row["completion_tokens"] == 12
    assert bool(row["error"]) == (stop != "end_turn" or not terminal)


def test_native_claude_structured_vision_and_explicit_budget(tmp_path):
    from llmx.providers import _openai_chat
    picture = tmp_path / "sample.png"
    picture.write_bytes(b"test image bytes")
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    def respond(request, **kwargs):
        body = json.loads(request.data)
        assert body["max_tokens"] == 128000
        assert body["output_config"] == {"effort": "max", "format": {"type": "json_schema", "schema": schema}}
        assert body["messages"][0]["content"][1]["source"]["media_type"] == "image/png"
        assert body["system"] == "system"
        return _native_sse_response({"model": "served", "stop_reason": "end_turn", "usage": {},
                                     "content": [{"type": "text", "text": '{"ok":true}'}]})
    with patch("urllib.request.urlopen", side_effect=respond), patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.usage_log.log_usage"):
        assert _openai_chat("hi", "claude-opus-5-5", "anthropic-direct", "system", .7, 30,
                            False, 64000, schema, "max", media=[picture]) == '{"ok":true}'


@pytest.mark.parametrize("request_id,expected", [("req_error-1", "req_error-1"), (None, None), ("private header\nvalue", None)])
def test_native_claude_http_error_never_falls_back_or_retries(request_id, expected):
    import urllib.error
    from llmx.providers import _openai_chat, RateLimitError
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("https://example.invalid", 429, "limit", {"request-id": request_id, "authorization": "private sentinel"}, None)) as call, patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.providers.OpenAI") as alternate, patch("llmx.usage_log.log_usage") as record:
        with pytest.raises(RateLimitError) as caught:
            _openai_chat("hi", "claude-opus-5-5", "anthropic-direct", None, .7, 30, False, None, None, "max")
    assert call.call_count == 1
    alternate.assert_not_called()
    assert record.call_args.kwargs["error"] == "Anthropic HTTP 429"
    assert caught.value.request_id == expected
    assert record.call_args.kwargs["request_id"] == expected
    assert record.call_args.kwargs["transport_error_class"] == "HTTPError"
    assert "private" not in str(record.call_args)


def test_native_public_response_retains_request_identity():
    from llmx.api import LLM
    response = _native_sse_response({"model": "served", "stop_reason": "end_turn", "usage": {}, "content": [{"type": "text", "text": "ok"}]})
    response.headers = {"request-id": "req_public-1"}
    with patch("urllib.request.urlopen", return_value=response), patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.api.check_api_key"), patch("llmx.spend_guard.enforce_daily_cap"), patch("llmx.usage_log.log_usage"):
        result = LLM(provider="anthropic-direct", model="claude-opus-5-5", auth="api").chat("hi")
    assert result.content == "ok"
    assert result.request_id == "req_public-1" and result.transport_phase == "complete"
    assert result.served_model == "served" and result.stop_reason == "end_turn"


@pytest.mark.parametrize("stop", ["end_turn", "max_tokens"])
def test_native_claude_structured_result_preserves_identity_on_failure(tmp_path, stop):
    from llmx.dispatch_api import dispatch
    payload = {"model": "actual-opus", "stop_reason": stop, "content": [{"type": "text", "text": "answer"}],
               "usage": {"input_tokens": 5, "output_tokens": 7, "new_input_tokens": 2}}
    with patch("urllib.request.urlopen", return_value=_native_sse_response(payload)), patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.api.check_api_key"), patch("llmx.spend_guard.enforce_daily_cap"), patch("llmx.usage_log.log_usage") as record:
        result = dispatch("hi", provider="anthropic-direct", model="claude-opus-5-5", auth="api", reasoning_effort="max")
    assert result.served_model == "actual-opus" and result.stop_reason == stop
    assert result.usage["completion_tokens"] == 7
    assert result.status == ("ok" if stop == "end_turn" else "model_error")
    if stop != "end_turn":
        assert result.text == "" and record.call_args.kwargs["error"]
        assert "[DEGRADED]" in result.usage["usage_note"]
    assert "[DEGRADED]" in record.call_args.kwargs["note"]
    assert "new_input_tokens" in record.call_args.kwargs["note"]


def test_dispatch_result_additions_preserve_positional_text():
    from llmx.dispatch_api import DispatchResult
    result = DispatchResult("ok", False, "existing positional text")
    assert result.text == "existing positional text"
    assert result.served_model is None and result.stop_reason is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure_kind", ["reset", "incomplete", "disconnect"])
def test_native_claude_read_errors_keep_transport_class(stream, failure_kind):
    import io
    import http.client
    from llmx.providers import _openai_chat, ServiceUnavailableError
    error = {"reset": ConnectionResetError("private read detail"),
             "incomplete": http.client.IncompleteRead(b"private partial"),
             "disconnect": http.client.RemoteDisconnected("private read detail")}[failure_kind]
    class BrokenResponse(io.BytesIO):
        def read(self, *args):
            raise error
        def __iter__(self):
            yield b'data: {"type":"message_start","message":{"model":"served","usage":{"input_tokens":10}}}\n'
            yield b'\n'
            raise error
    with patch("urllib.request.urlopen", return_value=BrokenResponse()) as native, patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.usage_log.log_usage") as record:
        with pytest.raises(ServiceUnavailableError) as caught:
            _openai_chat("hi", "claude-opus-5-5", "anthropic-direct", None, .7, 30, stream, None, None, None)
    assert native.call_count == 1
    assert "private" not in str(caught.value)
    assert record.call_args.kwargs["error"]
    assert record.call_args.kwargs["served_model"] == "served"


def test_native_claude_read_error_uses_existing_api_retry_owner():
    import io
    from llmx.api import LLM
    class BrokenResponse(io.BytesIO):
        def __iter__(self):
            raise ConnectionResetError("reset")
    success = {"model": "served", "stop_reason": "end_turn", "usage": {}, "content": [{"type": "text", "text": "ok"}]}
    with patch.dict(os.environ, {"LLMX_MAX_RETRIES": "2"}), patch("urllib.request.urlopen", side_effect=[BrokenResponse(), _native_sse_response(success)]) as native, patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.api.check_api_key"), patch("llmx.spend_guard.enforce_daily_cap"), patch("llmx.api.time.sleep") as sleep, patch("llmx.usage_log.log_usage") as record:
        response = LLM(provider="anthropic-direct", model="claude-opus-5-5", auth="api").chat("hi")
    assert response.content == "ok" and native.call_count == 2 and sleep.call_count == 1
    assert record.call_args_list[0].kwargs["error"]
    assert record.call_args_list[1].kwargs["error"] is None


@pytest.mark.parametrize("caller", ["chat", "dispatch"])
@pytest.mark.parametrize("exists", [False, True])
def test_public_native_claude_media_is_never_dropped(tmp_path, caller, exists):
    import base64
    from llmx.api import LLM
    from llmx.dispatch_api import dispatch
    picture = tmp_path / "pixel.png"
    pixels = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
    if exists:
        picture.write_bytes(pixels)
    def respond(request, **kwargs):
        blocks = json.loads(request.data)["messages"][0]["content"]
        assert base64.b64decode(blocks[1]["source"]["data"]) == pixels
        return _native_sse_response({"model": "served", "stop_reason": "end_turn", "usage": {}, "content": [{"type": "text", "text": "ok"}]})
    with patch("urllib.request.urlopen", side_effect=respond) as native, patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.api.check_api_key"), patch("llmx.spend_guard.enforce_daily_cap"), patch("llmx.usage_log.log_usage"):
        if caller == "dispatch":
            result = dispatch("image?", provider="anthropic-direct", model="claude-opus-5-5", auth="api", media=[picture])
            assert (result.status == "ok") is exists
        elif exists:
            assert LLM(provider="anthropic-direct", model="claude-opus-5-5", auth="api").chat("image?", media=[picture]).content == "ok"
        else:
            with pytest.raises(FileNotFoundError):
                LLM(provider="anthropic-direct", model="claude-opus-5-5", auth="api").chat("image?", media=[picture])
    assert native.call_count == int(exists)


def test_public_google_chat_forwards_supplied_media_to_existing_owner(tmp_path):
    from llmx.api import LLM
    picture = tmp_path / "pixel.png"
    with patch("llmx.api._google_chat", return_value="ok") as owner, patch("llmx.api.check_api_key"), patch("llmx.spend_guard.enforce_daily_cap"):
        assert LLM(provider="google", model="gemini-3.8-pro", auth="api").chat("image?", media=[picture]).content == "ok"
    assert owner.call_args.kwargs["media"] == [picture]


def test_native_haiku_keeps_model_cap_without_unsupported_effort():
    from llmx.providers import _openai_chat
    from llmx.dispatch_plan import resolve_effort
    with pytest.raises(ValueError, match="Unsupported native Claude effort"):
        resolve_effort("max", transport="anthropic-direct-api", provider="anthropic-direct", model="claude-haiku-4-5")
    def respond(request, **kwargs):
        body = json.loads(request.data)
        assert body["max_tokens"] == 64000
        assert "output_config" not in body
        return _native_sse_response({"model": "claude-haiku-4-5", "stop_reason": "end_turn", "usage": {}, "content": [{"type": "text", "text": "ok"}]})
    with patch("urllib.request.urlopen", side_effect=respond), patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.usage_log.log_usage"):
        assert _openai_chat("hi", "claude-haiku-4-5", "anthropic-direct", None, .7, 30, False, 100000, None, None) == "ok"


@pytest.mark.parametrize("model,total", [("claude-opus-5-5", 24192), ("claude-opus-5", 40192), ("claude-fable-5-1", 40192), ("claude-sonnet-5", 40192), ("claude-haiku-4-5", 8192)])
def test_native_claude_default_reserve_is_model_scoped(model, total):
    from llmx.providers import _openai_chat
    def respond(request, **kwargs):
        body = json.loads(request.data)
        assert body["max_tokens"] == total
        assert "output_config" not in body
        return _native_sse_response({"model": model, "stop_reason": "end_turn", "usage": {}, "content": [{"type": "text", "text": "ok"}]})
    with patch("urllib.request.urlopen", side_effect=respond), patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.usage_log.log_usage") as record:
        _openai_chat("hi", model, "anthropic-direct", None, .7, 30, False, None, None, None)
    assert record.call_args.kwargs["reasoning_effort"] is None
    assert "effort omitted" in record.call_args.kwargs["note"]


@pytest.mark.parametrize("status,kind", [(401, "ApiKeyError"), (402, "QuotaError"), (408, "TimeoutError_"), (429, "RateLimitError"), (500, "ServiceUnavailableError"), (400, "ModelError")])
def test_native_claude_http_errors_keep_public_classes(status, kind):
    import urllib.error
    from llmx import providers
    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("https://example.invalid", status, "private sentinel", {}, None)), patch("llmx.providers._get_api_key", return_value="fake"), patch("llmx.usage_log.log_usage") as record:
        with pytest.raises(getattr(providers, kind)) as caught:
            providers._openai_chat("hi", "claude-opus-5-5", "anthropic-direct", None, .7, 30, False, None, None, None)
    assert "private sentinel" not in str(caught.value)
    assert record.call_args.kwargs["error"] == f"Anthropic HTTP {status}"


@pytest.mark.parametrize("body,kind", [
    (b'{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}}', "QuotaError"),
    (b'{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: 999999 > 128000"}}', "ModelError"),
    (b"not json", "ModelError"),
])
def test_native_claude_unfunded_key_400_is_quota(body, kind):
    # 2026-10-24: a stale inherited key got 400 "credit balance is too low"; llmx reported a model error.
    import io
    import urllib.error

    from llmx import providers
    refusal = urllib.error.HTTPError("https://example.invalid", 400, "private sentinel", {}, io.BytesIO(body))
    with patch("urllib.request.urlopen", side_effect=refusal), patch("llmx.providers._get_api_key", return_value="fake"), \
            patch("llmx.usage_log.log_usage") as record, pytest.raises(getattr(providers, kind)) as caught:
        providers._openai_chat("hi", "claude-opus-5-5", "anthropic-direct", None, .7, 30, False, None, None, None)
    assert type(caught.value) is getattr(providers, kind)
    assert "private sentinel" not in str(caught.value)
    assert record.call_args.kwargs["error"] == "Anthropic HTTP 400"


@pytest.mark.parametrize("stream,models,expected", [
    (False, ["actual-model"], "actual-model"), (False, [None], None),
    (True, ["actual-model", "actual-model"], "actual-model"),
    (True, [None, None], None), (True, ["first-model", "second-model"], None),
    (True, ["actual-model", None], "actual-model"),
])
def test_openai_api_records_response_model_not_request(tmp_path, stream, models, expected):
    from llmx.providers import _openai_chat
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3)
    chunks = [SimpleNamespace(model=model, usage=usage, choices=[SimpleNamespace(
        delta=SimpleNamespace(content="OK"), finish_reason="stop")]) for model in models]
    response = SimpleNamespace(model=models[0], usage=usage, choices=[SimpleNamespace(
        message=SimpleNamespace(content="OK"), finish_reason="stop")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: iter(chunks) if stream else response)))
    ledger = tmp_path / "usage.jsonl"
    with patch("llmx.providers.OpenAI", return_value=client), patch(
        "llmx.providers._get_api_key", return_value="fake-key"
    ), patch("llmx.usage_log._LOG_PATH", ledger):
        _openai_chat("hi", "requested-model", "openai", None, 0.7, 30,
                     stream, None, None, None)
    row = json.loads(ledger.read_text())
    assert row["model"] == "requested-model"
    assert row["served_model"] == expected
    assert row["prompt_tokens"] == 10 and row["completion_tokens"] == 3
    assert row["completion_includes_reasoning"] is True
    if expected is None:
        assert "[DEGRADED]" in row["note"] and "served_model" in row["note"]


def _claude_verbose_stdout(
    result: str,
    *,
    text_blocks: list[str] | None = None,
    usage: dict | None = None,
    model_usage: dict | None = None,
) -> str:
    return json.dumps(
        [
            {
                "type": "assistant",
                "message": {
                    "id": "msg-final",
                    "content": [
                        {"type": "text", "text": text}
                        for text in (text_blocks if text_blocks is not None else [result])
                    ],
                },
            },
            {
                "type": "result",
                "is_error": False,
                "result": result,
                "usage": usage or {},
                "modelUsage": model_usage or {},
            },
        ]
    )


def _claude_events_stdout(
    blocks: list[tuple[str, str, str]],
    *,
    result: str,
    result_fields: dict | None = None,
) -> str:
    """Verbose Claude JSON in the live shape: one assistant event per content block."""
    events: list[dict] = [
        {
            "type": "assistant",
            "message": {
                "id": message_id,
                "stop_reason": None,
                "content": [{"type": block_type, block_type: text}],
            },
        }
        for message_id, block_type, text in blocks
    ]
    events.append(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": result,
            "stop_reason": "end_turn",
            "terminal_reason": "completed",
            "num_turns": 1,
            **(result_fields or {}),
        }
    )
    return json.dumps(events)


def _write_rollout(path: Path, input_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "event_msg", "payload": {"type": "irrelevant"}})
        + "\n"
        + json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": input_tokens,
                            "cached_input_tokens": 3,
                            "output_tokens": 5,
                            "reasoning_output_tokens": 2,
                            "total_tokens": input_tokens + 5,
                        }
                    },
                },
            }
        )
        + "\n"
    )


def _write_turns(
    path: Path, turns: list[tuple[int, int, int]], *, with_totals: bool = True
) -> None:
    """One token_count event per (input, output, reasoning) turn, with running totals."""
    path.parent.mkdir(parents=True, exist_ok=True)
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_output_tokens": 0}
    lines = []
    for input_tokens, generated_tokens, reasoning_tokens in turns:
        turn = {
            "input_tokens": input_tokens,
            "cached_input_tokens": 0,
            "output_tokens": generated_tokens,
            "reasoning_output_tokens": reasoning_tokens,
        }
        for key, value in turn.items():
            totals[key] += value
        info = {"last_token_usage": turn}
        if with_totals:
            info["total_token_usage"] = dict(totals)
        lines.append(json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": info}}))
    path.write_text("\n".join(lines) + "\n")


_OURS = "01a0e8c2-1c3c-7cf3-814f-a9ba9ee4657e"
_OTHER = "01a0e8c2-1c3c-78d3-826a-e70715a36db1"


class TestCodexRolloutUsage(unittest.TestCase):
    def test_reads_rollout_named_by_session_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(root / "2026" / "07" / "06" / f"rollout-2026-07-06T01-03-50-{_OURS}.jsonl", 11)

            usage, note = _latest_codex_rollout_usage({}, started_at=0, session_id=_OURS, root=root)

        self.assertIsNone(note)
        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertEqual(usage["cached_tokens"], 3)
        self.assertEqual(usage["completion_tokens"], 5)
        self.assertEqual(usage["reasoning_tokens"], 2)

    def test_multi_turn_rollout_reports_the_call_total(self):
        # 2026-10-08: agent-mode diff hunts logged only their closing turn (~450 output
        # tokens each) because the reader preferred last_token_usage to the running total.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "2026" / "10" / "08" / f"rollout-2026-10-08T01-03-50-{_OURS}.jsonl"
            _write_turns(path, [(100, 40, 30), (150, 10, 5)])

            usage, note = _latest_codex_rollout_usage({}, started_at=0, session_id=_OURS, root=root)

        self.assertIsNone(note)
        self.assertEqual(usage["prompt_tokens"], 250)
        self.assertEqual(usage["completion_tokens"], 50)
        self.assertEqual(usage["reasoning_tokens"], 35)

    def test_multi_turn_rollout_without_totals_says_counts_are_the_last_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "2026" / "10" / "08" / f"rollout-2026-10-08T01-03-50-{_OURS}.jsonl"
            _write_turns(path, [(100, 40, 30), (150, 10, 5)], with_totals=False)

            usage, note = _latest_codex_rollout_usage({}, started_at=0, session_id=_OURS, root=root)

        self.assertEqual(usage["prompt_tokens"], 150)
        self.assertIn("last turn's only", note)

    def test_parallel_calls_each_read_their_own_rollout(self):
        # 2026-09-29: parallel codex calls logged the newest rollout, which belonged to
        # another call (still running, so sometimes null tokens).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "09" / "29"
            ours = day / f"rollout-2026-09-29T01-03-50-{_OURS}.jsonl"
            other = day / f"rollout-2026-09-29T01-03-50-{_OTHER}.jsonl"
            _write_rollout(ours, 11)
            _write_rollout(other, 99)
            later = ours.stat().st_mtime_ns + 1_000_000_000
            os.utime(other, ns=(later, later))

            usage, note = _latest_codex_rollout_usage({}, started_at=0, session_id=_OURS, root=root)

        self.assertIsNone(note)
        self.assertEqual(usage["prompt_tokens"], 11)

    def test_without_session_id_several_new_rollouts_leave_usage_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026" / "09" / "29"
            _write_rollout(day / f"rollout-2026-09-29T01-03-50-{_OURS}.jsonl", 11)
            _write_rollout(day / f"rollout-2026-09-29T01-03-50-{_OTHER}.jsonl", 99)

            usage, note = _latest_codex_rollout_usage({}, started_at=0, root=root)

        self.assertIsNone(usage["prompt_tokens"])
        self.assertIn("2 new codex rollouts", note)

    def test_without_session_id_single_new_rollout_is_attributed_by_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rollout(root / "2026" / "09" / "29" / f"rollout-2026-09-29T01-03-50-{_OURS}.jsonl", 11)

            usage, note = _latest_codex_rollout_usage({}, started_at=0, root=root)

        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertIn("attributed by time", note)

    def test_changed_parent_rollout_is_never_attributed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "2026" / "09" / "29" / f"rollout-2026-09-29T00-00-00-{_OTHER}.jsonl"
            _write_rollout(parent, 99)
            before = {parent: parent.stat().st_mtime_ns - 1}

            usage, note = _latest_codex_rollout_usage(before, started_at=0, root=root)

        self.assertIsNone(usage["prompt_tokens"])
        self.assertIn("0 new codex rollouts", note)

    def test_session_id_without_rollout_leaves_usage_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            usage, note = _latest_codex_rollout_usage({}, started_at=0, session_id=_OURS, root=Path(tmp))

        self.assertIsNone(usage["prompt_tokens"])
        self.assertIn(f"0 codex rollouts match session {_OURS}", note)

    def test_session_id_parsed_from_stderr_header(self):
        stderr = (
            "Reading additional input from stdin...\n"
            "OpenAI Codex v0.156.1\n--------\nworkdir: /tmp/x\nmodel: gpt-6-astra\n"
            "session id: 01A0E8D3-D4D8-7530-8ACC-D342521484F1\n--------\nuser\nhi\n"
        )
        self.assertEqual(_codex_session_id(stderr), "01a0e8d3-d4d8-7530-8acc-d342521484f1")
        self.assertIsNone(_codex_session_id("user\nsession id: not-an-id\n"))
        self.assertIsNone(_codex_session_id(None))

    def test_missing_rollout_returns_null_usage_with_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            usage, note = _latest_codex_rollout_usage(
                {},
                started_at=0,
                root=Path(tmp) / "missing",
            )

        self.assertTrue(note)
        self.assertIsNone(usage["prompt_tokens"])
        self.assertIsNone(usage["completion_tokens"])
        self.assertIsNone(usage["reasoning_tokens"])


class TestClaudeCliUsage(unittest.TestCase):
    def test_parse_claude_json_keeps_absent_reasoning_null(self):
        stdout = _claude_verbose_stdout(
            "ok",
            usage={
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 4,
            },
            model_usage={"claude-fable-5[1m]": {}},
        )

        result, usage = _parse_claude_json(stdout)

        self.assertEqual(result, "ok")
        self.assertEqual(usage["model"], "claude-fable-5")
        self.assertEqual(usage["served_model"], "claude-fable-5")
        self.assertIsNone(usage["reasoning_tokens"])

    def test_claude_log_receipt_never_falls_back_to_requested_model(self):
        # 2026-09-30: model mixed requested/provider values and served_model
        # stayed null, so the old telemetry could not verify model provenance.
        for model_usage, expected in (
            ({"claude-opus-5-5[1m]": {}}, "claude-opus-5-5"),
            ({}, None),
            ({"claude-opus-5-5": {}, "claude-haiku-4-5": {}}, None),
        ):
            with self.subTest(model_usage=model_usage), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "usage.jsonl"
                with (
                    patch("llmx.cli_backends.subprocess.Popen") as popen,
                    patch("llmx.usage_log._LOG_PATH", path),
                    patch("llmx.usage_log._resolve_caller", return_value="offline-test"),
                ):
                    process = popen.return_value
                    process.pid = 123
                    process.returncode = 0
                    process.communicate.return_value = (
                        _claude_verbose_stdout("ok", model_usage=model_usage), ""
                    )
                    result = cli_chat(
                        "claude-cli", "hi", "requested-alias", 30,
                        reasoning_effort="xhigh",
                    )
                row = json.loads(path.read_text())
                command = next(
                    c.args[0] for c in popen.call_args_list
                    if c.args and c.args[0][0] == "claude"
                )
                self.assertEqual(result, "ok")
                self.assertEqual(command[command.index("--effort") + 1], "xhigh")
                self.assertEqual(row["reasoning_effort"], "xhigh")
                self.assertEqual(row["served_model"], expected)
                self.assertEqual(row["source"], "claude-cli-json")
                if expected is None:
                    self.assertIn("served_model unknown", row["note"])
                if not model_usage:
                    self.assertEqual(row["model"], "requested-alias")

    def test_parse_claude_json_reads_reasoning_when_exposed(self):
        stdout = _claude_verbose_stdout(
            "ok",
            usage={
                "input_tokens": 10,
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        )

        _, usage = _parse_claude_json(stdout)

        self.assertEqual(usage["reasoning_tokens"], 5)

    def test_parse_claude_json_preserves_reported_zero_reasoning(self):
        stdout = _claude_verbose_stdout(
            "ok",
            usage={
                "input_tokens": 10,
                "output_tokens": 2,
                "reasoning_tokens": 0,
            },
        )

        _, usage = _parse_claude_json(stdout)

        self.assertEqual(usage["reasoning_tokens"], 0)

    def test_parse_claude_json_returns_full_message_when_result_keeps_last_block(self):
        # 2026-10-15: an Opus 5.5 review (14,657 chars) was refused because `result` held only
        # its final block. The verbose events of a completed single-turn message hold the answer.
        stdout = _claude_events_stdout(
            [("msg-final", "thinking", "plan"), ("msg-final", "text", "PREFIX"),
             ("msg-final", "thinking", "more"), ("msg-final", "text", "TAIL")],
            result="TAIL",
        )

        result, usage = _parse_claude_json(stdout)

        self.assertEqual(result, "PREFIXTAIL")
        self.assertEqual(usage["result_projection"], "final_text_block")

    def test_parse_claude_json_compares_the_last_block_without_surrounding_whitespace(self):
        stdout = _claude_events_stdout(
            [("msg-final", "text", "PREFIX\n"), ("msg-final", "text", "\nTAIL\n")],
            result="TAIL",
        )

        result, usage = _parse_claude_json(stdout)

        self.assertEqual(result, "PREFIX\n\nTAIL\n")
        self.assertEqual(usage["result_projection"], "final_text_block")

    def test_parse_claude_json_returns_untrimmed_text_when_result_was_trimmed(self):
        stdout = _claude_events_stdout([("msg-final", "text", "answer\n")], result="answer")

        result, usage = _parse_claude_json(stdout)

        self.assertEqual(result, "answer\n")
        self.assertEqual(usage["result_projection"], "trimmed_whitespace")

    def test_parse_claude_json_refuses_last_block_projection_across_messages(self):
        # A second message means a tool turn or continuation may have split the answer.
        stdout = _claude_events_stdout(
            [("msg-a", "text", "FIRST HALF"), ("msg-b", "text", "PREFIX"), ("msg-b", "text", "TAIL")],
            result="TAIL",
        )

        result, usage = _parse_claude_json(stdout)

        self.assertIsInstance(result, CliBackendFailure)
        self.assertEqual(result.kind, LlmxError)
        self.assertIn("omitted assistant text blocks", result.detail)
        self.assertIn("reconstructed_chars=10", result.detail)
        self.assertIn("result_chars=4", result.detail)
        self.assertIn("omitted_chars=6", result.detail)
        self.assertIsNone(usage)

    def test_parse_claude_json_refuses_last_block_projection_after_a_length_stop(self):
        for result_fields in ({"stop_reason": "max_tokens"}, {"terminal_reason": "max_turns"}, {"num_turns": 2}):
            with self.subTest(result_fields=result_fields):
                stdout = _claude_events_stdout(
                    [("msg-final", "text", "PREFIX"), ("msg-final", "text", "TAIL")],
                    result="TAIL",
                    result_fields=result_fields,
                )

                result, usage = _parse_claude_json(stdout)

                self.assertIsInstance(result, CliBackendFailure)
                self.assertIsNone(usage)

    def test_parse_claude_json_refusal_saves_the_stream_shape(self):
        stdout = _claude_events_stdout(
            [("msg-final", "thinking", "secret plan"), ("msg-final", "text", "PREFIX"),
             ("msg-final", "text", "TAIL")],
            result="OTHER",
        )

        result, usage = _parse_claude_json(stdout)

        self.assertIsInstance(result, CliBackendFailure)
        self.assertIn("disagreed with reconstructed assistant text", result.detail)
        saved = Path(result.detail.split("stream shape saved to ")[1])
        record = json.loads(saved.read_text())
        self.assertEqual(record["result"], "OTHER")
        blocks = [block for event in record["events"] for block in event["blocks"]]
        self.assertEqual([block.get("text") for block in blocks if block["type"] == "text"], ["PREFIX", "TAIL"])
        thinking = [block for block in blocks if block["type"] == "thinking"]
        self.assertEqual(thinking, [{"type": "thinking", "chars": len("secret plan")}])

    def test_parse_claude_json_refusal_reports_an_unsaved_stream(self):
        stdout = _claude_events_stdout(
            [("msg-final", "text", "PREFIX"), ("msg-final", "text", "TAIL")], result="OTHER"
        )
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "file"
            blocker.write_text("")
            with patch("llmx.cli_backends.CLAUDE_REFUSED_STREAM_DIR", blocker / "dir"):
                result, _ = _parse_claude_json(stdout)

        self.assertIsInstance(result, CliBackendFailure)
        self.assertIn("[DEGRADED] stream shape not saved", result.detail)

    def test_parse_claude_json_accepts_complete_multiblock_projection(self):
        stdout = _claude_verbose_stdout(
            "PREFIXTAIL",
            text_blocks=["PREFIX", "TAIL"],
        )

        result, usage = _parse_claude_json(stdout)

        self.assertEqual(result, "PREFIXTAIL")
        self.assertIsInstance(usage, dict)

    def test_parse_claude_json_accepts_complete_single_block_projection(self):
        stdout = _claude_verbose_stdout("complete response")

        result, usage = _parse_claude_json(stdout)

        self.assertEqual(result, "complete response")
        self.assertIsInstance(usage, dict)

    def test_live_payload_logs_total_input_and_cache_write(self):
        # 2026-10-12: rows logged prompt_tokens=2 (Anthropic input_tokens, which
        # excludes cache) and dropped cache_creation_input_tokens. Fixture is a
        # real `claude -p --output-format json --verbose` opus-5-5 result.
        stdout = (FIXTURES / "claude_cli_usage_2026-10-12.json").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            with (
                patch("llmx.cli_backends.subprocess.Popen") as popen,
                patch("llmx.usage_log._LOG_PATH", path),
                patch("llmx.usage_log._resolve_caller", return_value="offline-test"),
            ):
                process = popen.return_value
                process.pid = 123
                process.returncode = 0
                process.communicate.return_value = (stdout, "")
                cli_chat("claude-cli", "hi", "claude-opus-5-5", 30)
            row = json.loads(path.read_text())

        self.assertEqual(row["prompt_tokens"], 2 + 9738 + 15550)
        self.assertEqual(row["cached_tokens"], 9738)
        self.assertEqual(row["cache_write_tokens"], 15550)
        self.assertEqual(row["completion_tokens"], 110)
        self.assertEqual(row["reasoning_tokens"], 107)
        # Claude output_tokens already includes thinking; never add it twice.
        self.assertTrue(row["completion_includes_reasoning"])
        self.assertEqual(output_tokens(row), 110)
        self.assertEqual(row["reported_cost_usd"], 0.1285556)
        self.assertIsNone(row["note"])
        # Without the CLI's figure, Anthropic's cache rates reproduce it from the counts.
        from llmx.usage_report import cost_for_usage

        self.assertAlmostEqual(cost_for_usage({**row, "reported_cost_usd": None}), 0.1285556)

    def test_usage_shape_drift_is_noted_not_silent(self):
        for usage, expected in (
            ({"input_tokens": 2, "output_tokens": 4}, "cache_creation_input_tokens"),
            (
                {
                    "input_tokens": 2, "output_tokens": 4,
                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                    "cache_storage_tokens": 9,
                },
                "unrecognized usage token fields ['cache_storage_tokens']",
            ),
        ):
            with self.subTest(usage=usage):
                _, parsed = _parse_claude_json(
                    _claude_verbose_stdout("ok", usage=usage, model_usage={"m": {}})
                )
                self.assertIn("[DEGRADED] claude-cli usage", parsed["usage_note"])
                self.assertIn(expected, parsed["usage_note"])


class TestOpenRouterUsage(unittest.TestCase):
    def test_cache_write_usage_is_preserved_and_missing_stays_unknown(self):
        raw = {
            "input_tokens": 15_000,
            "output_tokens": 100,
            "input_tokens_details": {"cached_tokens": 12_000, "cache_write_tokens": 3_000},
        }
        usage = _normalize_usage("openai", raw)
        self.assertEqual(usage["cache_write_tokens"], 3_000)
        self.assertTrue(usage["completion_includes_reasoning"])
        del raw["input_tokens_details"]["cache_write_tokens"]
        self.assertIsNone(_normalize_usage("openai", raw)["cache_write_tokens"])

    def test_normalize_usage_accepts_dict_reasoning_details(self):
        usage = _normalize_usage(
            "openrouter",
            {
                "prompt_tokens": 13,
                "completion_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 6},
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        )

        self.assertEqual(usage["prompt_tokens"], 13)
        self.assertEqual(usage["completion_tokens"], 8)
        self.assertEqual(usage["total_tokens"], 21)
        self.assertEqual(usage["reasoning_tokens"], 6)
        self.assertEqual(usage["cached_tokens"], 4)

    def test_normalize_usage_accepts_reasoning_variant(self):
        usage = _normalize_usage(
            "openrouter",
            {
                "input_tokens": 13,
                "output_tokens": 8,
                "output_tokens_details": {"reasoning": 6},
                "input_tokens_details": {"cached_input_tokens": 4},
            },
        )

        self.assertEqual(usage["prompt_tokens"], 13)
        self.assertEqual(usage["completion_tokens"], 8)
        self.assertEqual(usage["reasoning_tokens"], 6)
        self.assertEqual(usage["cached_tokens"], 4)


class TestAstraCost(unittest.TestCase):
    def test_report_and_guard_agree_when_cache_writes_are_unreported(self):
        from datetime import datetime, timezone

        from llmx.spend_guard import metered_spend_today
        from llmx.usage_report import est_cost, summarize

        now = datetime.now(timezone.utc)
        row = {
            "ts": now.isoformat(), "model": "gpt-6-astra", "transport": "api",
            "prompt_tokens": 300_000, "completion_tokens": 10_000,
            "cached_tokens": 0, "cache_write_tokens": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(metered_spend_today(path), (8.25, True))
            report = summarize(by="model", since=now.date().isoformat(), log=path)
        self.assertIn("$8.25", report)
        self.assertIn("conservative estimate", report)
        self.assertIn("cache-write rate", report)
        self.assertEqual(est_cost("gpt-6-astra", 300_000, 10_000), 6.75)

    def test_unreported_totals_are_unknown_but_explicit_zero_is_known(self):
        from llmx.usage_report import cost_for_usage

        complete = {"model": "gpt-6-astra", "prompt_tokens": 0, "completion_tokens": 0}
        self.assertEqual(cost_for_usage(complete), 0.0)
        for field in ("prompt_tokens", "completion_tokens"):
            with self.subTest(field=field):
                row = {**complete, field: None}
                self.assertIsNone(cost_for_usage(row))
                del row[field]
                self.assertIsNone(cost_for_usage(row))

    def test_rollup_marks_missing_usage_unknown_in_group_and_total(self):
        from llmx.usage_report import summarize

        row = {
            "ts": "2026-09-05T00:00:00Z", "model": "gpt-6-astra", "provider": "openai",
            "prompt_tokens": None, "completion_tokens": None, "error": "GeneratorExit: ",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            path.write_text(json.dumps(row) + "\n")
            report = summarize(by="model", since="2026-09-05", log=path)
        self.assertEqual(report.count("$0.00+?"), 2)
        self.assertIn("unreported usage", report)

    def test_usage_log_preserves_cache_and_output_semantics_for_pricing(self):
        from llmx.usage_log import log_usage
        from llmx.usage_report import cost_for_usage

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            with patch("llmx.usage_log._LOG_PATH", path), patch(
                "llmx.usage_log._resolve_caller", return_value="offline-test"
            ):
                log_usage(
                    provider="openai", model="gpt-6-astra", transport="api",
                    reasoning_effort="max", prompt_tokens=300_000,
                    completion_tokens=10_000, reasoning_tokens=8_000,
                    cached_tokens=200_000, cache_write_tokens=50_000,
                    completion_includes_reasoning=True, latency_s=0.1,
                )
            row = json.loads(path.read_text())
        self.assertEqual(row["cache_write_tokens"], 50_000)
        self.assertTrue(row["completion_includes_reasoning"])
        self.assertAlmostEqual(cost_for_usage(row), 3.40)

    def test_pricing_boundaries_and_disjoint_cache_categories(self):
        from llmx.usage_report import est_cost

        cases = [
            (272_000, 10_000, 0, 0, 3.22),
            (272_001, 10_000, 0, 0, 6.19002),
            (300_000, 10_000, 0, 0, 6.75),
            (100_000, 10_000, 80_000, 0, 0.78),
            (100_000, 0, 0, 100_000, 1.25),
            (300_000, 10_000, 200_000, 50_000, 3.40),
        ]
        for model in ("gpt-6-astra", "gpt-6"):
            for prompt, output, cached, written, expected in cases:
                with self.subTest(model=model, prompt=prompt, cached=cached, written=written):
                    self.assertAlmostEqual(est_cost(
                        model, prompt, output, cached_tokens=cached, cache_write_tokens=written,
                    ), expected)
        self.assertEqual(est_cost("gpt-5.6-luna", 1_000_000, 1_000_000), 1.40)
        self.assertIsNone(est_cost("unknown", 1, 1))

    def test_claude_cache_rates_and_reported_cost(self):
        from llmx.usage_report import cost_for_usage, est_cost

        # Reads cost 10% of input, except Opus 5.5 (5%) and Fable 5.1 (2.5%).
        for model, expected in (
            ("claude-fable-5-1", 0.25), ("claude-opus-5-5", 0.20), ("claude-sonnet-5", 0.20),
        ):
            with self.subTest(model=model):
                self.assertAlmostEqual(est_cost(
                    model, 1_000_000, 0, cached_tokens=1_000_000, cache_write_tokens=0,
                ), expected)
        # Writes are 1-hour entries (claude-cli), 2x input.
        self.assertAlmostEqual(est_cost(
            "claude-opus-5-5", 1_000_000, 0, cached_tokens=0, cache_write_tokens=1_000_000,
        ), 8.0)
        self.assertAlmostEqual(est_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000), 6.0)
        # A live claude-cli row (2026-10-12): the estimate matches the CLI's own figure.
        row = {
            "model": "claude-opus-5-5", "prompt_tokens": 25_181, "completion_tokens": 4,
            "cached_tokens": 7_944, "cache_write_tokens": 17_235,
        }
        self.assertAlmostEqual(cost_for_usage(row), 0.1395568)
        self.assertEqual(cost_for_usage({**row, "reported_cost_usd": 0.5}), 0.5)
        unreported = {"model": "?", "prompt_tokens": None, "completion_tokens": None}
        self.assertEqual(cost_for_usage({**unreported, "reported_cost_usd": 0.01}), 0.01)

    def test_reasoning_subset_is_not_added_twice(self):
        from llmx.usage_report import cost_for_usage, output_tokens

        row = {
            "model": "gpt-6-astra", "provider": "openai",
            "prompt_tokens": 10_000, "completion_tokens": 10_000, "reasoning_tokens": 8_000,
        }
        self.assertEqual(output_tokens(row), 10_000)
        self.assertAlmostEqual(cost_for_usage(row), 0.60)
        row.update(provider="google", model="gemini-3.8-flash", completion_includes_reasoning=False)
        self.assertEqual(output_tokens(row), 18_000)
        row.update(provider="openrouter", model="qwen/qwen3.8-27b", completion_includes_reasoning=True)
        self.assertEqual(output_tokens(row), 10_000)

    def test_rollup_prices_request_rows_before_aggregation(self):
        from llmx.usage_report import summarize

        row = {
            "ts": "2026-09-05T00:00:00Z", "model": "gpt-6-astra", "provider": "openai",
            "prompt_tokens": 100_000, "completion_tokens": 10_000, "reasoning_tokens": 8_000,
            "cached_tokens": 80_000, "cache_write_tokens": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage.jsonl"
            path.write_text((json.dumps(row) + "\n") * 3)
            report = summarize(by="model", since="2026-09-05", log=path)
        self.assertIn("2.34", report)


if __name__ == "__main__":
    unittest.main()


def test_client_rejected_call_costs_zero_not_unknown() -> None:
    from llmx.usage_report import cost_for_usage

    rejected = {
        "model": "gpt-6-astra",
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": "RateLimitError: Error code: 429 - {'error': {'code': 'credit_balance_exhausted'}}",
    }
    assert cost_for_usage(rejected, conservative=True) == 0.0
    timed_out = {**rejected, "error": "APITimeoutError: Request timed out."}
    assert cost_for_usage(timed_out, conservative=True) is None
