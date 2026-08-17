"""OpenRouter reasoning-effort wire format (offline).

Regression gate for the 2026-08-17 silent flag-drop: `-e medium` on
qwen/qwen3.8-27b never reached OpenRouter because the model is absent from
MODEL_RESTRICTIONS, so providers.chat nulled the effort out. The model then ran
at its default maximum reasoning and burned 103,640 completion tokens as
thinking with zero content across three probes.

These tests assert the OUTGOING REQUEST BODY, both polarities: an effort that is
requested must appear, and an effort that is not requested must leave no trace.
Every test patches the SDK client, so no request leaves the machine and nothing
is billed. The one SDK error type needed for the fail-loud test is taken from
llmx.providers (which owns the `openai` handle) rather than imported raw here.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llmx.dispatch_plan import map_effort_for_backend
from llmx.providers import openrouter_reasoning_body


def _fake_response(content: str = "OK"):
    """Minimal stand-in for an OpenAI SDK ChatCompletion."""
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
    )
    message = SimpleNamespace(content=content, refusal=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage)


class _CapturingClient:
    """Records the kwargs handed to chat.completions.create."""

    def __init__(self, calls: list, error: Exception | None = None):
        self._calls = calls
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self._calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return _fake_response()


class _Capture:
    """Context manager: patch the SDK client + usage log, collect request bodies."""

    def __init__(self, error: Exception | None = None):
        self.calls: list = []
        self._error = error
        self._patches: list = []

    def __enter__(self):
        self._patches = [
            patch(
                "llmx.providers.OpenAI",
                lambda **_: _CapturingClient(self.calls, self._error),
            ),
            patch("llmx.providers._get_api_key", return_value="test-key"),
            patch("llmx.providers.check_api_key", return_value=None),
            patch("llmx.spend_guard.enforce_daily_cap", return_value=None),
            patch("llmx.usage_log.log_usage", return_value=None),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False

    @property
    def body(self) -> dict:
        if len(self.calls) != 1:
            raise AssertionError(f"expected exactly 1 API call, got {len(self.calls)}")
        return self.calls[0]

    @property
    def reasoning(self):
        """The `reasoning` value on the wire, or None if no field was sent."""
        return self.body.get("extra_body", {}).get("reasoning")


def _chat(provider, model, effort, temperature=0.7):
    """Drive the FULL dispatch path (llmx.providers.chat), not just the builder.

    The regression lived upstream of the request builder, so a test that called
    _openai_chat directly would have passed even with the bug present.
    """
    from llmx import providers

    return providers.chat(
        prompt="hi",
        provider=provider,
        model=model,
        temperature=temperature,
        reasoning_effort=effort,
        stream=False,
        debug=False,
        json_output=False,
        timeout=30,
    )


def _chat_openrouter(effort, model="qwen/qwen3.8-27b"):
    return _chat("openrouter", model, effort)


class TestOpenrouterReasoningBody(unittest.TestCase):
    """Unit-level mapping table."""

    def test_effort_tiers_pass_through(self):
        for token in ("low", "medium", "high"):
            self.assertEqual(
                openrouter_reasoning_body(token), {"effort": token}, msg=token
            )

    def test_minimal_passes_through(self):
        # Verified accepted live 2026-08-17 (no 400 from OpenRouter).
        self.assertEqual(openrouter_reasoning_body("minimal"), {"effort": "minimal"})

    def test_beyond_ceiling_maps_to_high(self):
        for token in ("xhigh", "max"):
            self.assertEqual(
                openrouter_reasoning_body(token), {"effort": "high"}, msg=token
            )

    def test_none_disables(self):
        self.assertEqual(openrouter_reasoning_body("none"), {"enabled": False})

    def test_unset_sends_nothing(self):
        self.assertIsNone(openrouter_reasoning_body(None))
        self.assertIsNone(openrouter_reasoning_body(""))

    def test_unknown_token_raises_rather_than_drops(self):
        with self.assertRaises(ValueError):
            openrouter_reasoning_body("turbo")

    def test_returned_body_is_not_the_shared_table_entry(self):
        body = openrouter_reasoning_body("low")
        body["effort"] = "mutated"
        self.assertEqual(openrouter_reasoning_body("low"), {"effort": "low"})


class TestOpenrouterRequestBody(unittest.TestCase):
    """Both polarities of the outgoing body, through the full dispatch path."""

    def test_low_carries_effort_low(self):
        with _Capture() as cap:
            _chat_openrouter("low")
        self.assertEqual(cap.reasoning, {"effort": "low"})

    def test_medium_carries_effort_medium(self):
        # The exact call that silently no-op'd on 2026-08-17.
        with _Capture() as cap:
            _chat_openrouter("medium")
        self.assertEqual(cap.reasoning, {"effort": "medium"})

    def test_none_carries_enabled_false(self):
        with _Capture() as cap:
            _chat_openrouter("none")
        self.assertEqual(cap.reasoning, {"enabled": False})

    def test_max_carries_effort_high(self):
        with _Capture() as cap:
            _chat_openrouter("max")
        self.assertEqual(cap.reasoning, {"effort": "high"})

    def test_unset_sends_no_reasoning_key(self):
        """Compatibility invariant: no -e ⇒ byte-identical to the old default."""
        with _Capture() as cap:
            _chat_openrouter(None)
        self.assertNotIn("extra_body", cap.body)
        self.assertNotIn("reasoning", cap.body)
        self.assertNotIn("reasoning_effort", cap.body)

    def test_unset_sends_no_reasoning_key_on_table_matching_model(self):
        """A substring hit in MODEL_RESTRICTIONS must not inject an effort here.

        get_model_restriction matches by substring, so `openai/gpt-5.6` served by
        OpenRouter would otherwise inherit the OpenAI table's default_effort.
        """
        with _Capture() as cap:
            _chat_openrouter(None, model="openai/gpt-5.6")
        self.assertNotIn("extra_body", cap.body)
        self.assertNotIn("reasoning_effort", cap.body)

    def test_openrouter_never_uses_the_top_level_string_channel(self):
        with _Capture() as cap:
            _chat_openrouter("high")
        self.assertNotIn("reasoning_effort", cap.body)

    def test_api_error_on_reasoning_field_is_not_silently_retried(self):
        """Fail loud: no retry that drops the field the caller asked for."""
        from llmx.providers import LlmxError, openai_module

        error = openai_module.APIStatusError(
            "400 unsupported parameter: reasoning",
            response=SimpleNamespace(status_code=400, headers={}, request=None),
            body={"error": {"message": "unsupported parameter: reasoning"}},
        )
        with _Capture(error=error) as cap:
            with self.assertRaises(LlmxError) as ctx:
                _chat_openrouter("low")
        self.assertEqual(len(cap.calls), 1)  # one attempt, no field-dropping retry
        self.assertIn("reasoning", str(ctx.exception))


class TestOtherTransportsUnchanged(unittest.TestCase):
    """The fix is scoped to openrouter; nothing else may move."""

    def test_openai_still_uses_top_level_reasoning_effort(self):
        with _Capture() as cap:
            _chat("openai", "gpt-5.6", "low", temperature=1.0)
        self.assertEqual(cap.body.get("reasoning_effort"), "low")
        self.assertNotIn("extra_body", cap.body)

    def test_unsupported_model_effort_still_dropped_off_openrouter(self):
        """The MODEL_RESTRICTIONS gate still governs every other provider."""
        with _Capture() as cap:
            _chat("deepseek", "deepseek-chat", "low")
        self.assertNotIn("reasoning_effort", cap.body)
        self.assertNotIn("extra_body", cap.body)


class TestPlanReportsWhatTheWireCarries(unittest.TestCase):
    """effort_applied is what the dispatch stderr line prints — keep it truthful."""

    def test_max_reported_as_high(self):
        applied, warns = map_effort_for_backend(
            "max", transport="openrouter-api", provider="openrouter"
        )
        self.assertEqual(applied, "high")
        self.assertTrue(warns)

    def test_medium_reported_verbatim(self):
        applied, warns = map_effort_for_backend(
            "medium", transport="openrouter-api", provider="openrouter"
        )
        self.assertEqual(applied, "medium")
        self.assertFalse(warns)

    def test_none_survives_to_the_backend(self):
        applied, _ = map_effort_for_backend(
            "none", transport="openrouter-api", provider="openrouter"
        )
        self.assertEqual(applied, "none")

    def test_openai_max_mapping_untouched(self):
        applied, _ = map_effort_for_backend(
            "max", transport="openai-api", provider="openai", model="gpt-5.4"
        )
        self.assertEqual(applied, "xhigh")


if __name__ == "__main__":
    unittest.main()
