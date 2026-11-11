"""Google ClientError -> typed llmx error mapping (providers.chat google branch)."""

import unittest
from unittest.mock import patch

from google.genai import errors as genai_errors

from llmx import providers
from llmx.providers import QuotaError, RateLimitError

SPEND_CAP_MSG = (
    "Your project has exceeded its monthly spending cap. Please go to AI Studio at "
    "https://ai.studio/spend to manage your project spend cap."
)


def _client_error(message: str) -> genai_errors.ClientError:
    return genai_errors.ClientError(
        429, {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED"}}
    )


def _call(exc: Exception):
    with patch.object(providers, "_google_chat", side_effect=exc):
        return providers.chat(
            prompt="hi",
            provider="google",
            model="gemini-4-argon",
            temperature=0.0,
            reasoning_effort=None,
            stream=False,
            debug=False,
            json_output=False,
        )


class GoogleClientErrorMappingTest(unittest.TestCase):
    def test_gemini_spend_cap_client_error_is_quota(self) -> None:
        # 2026-10-20 incident: monthly spending cap arrives as 429 RESOURCE_EXHAUSTED.
        with self.assertRaises(QuotaError) as ctx:
            _call(_client_error(SPEND_CAP_MSG))
        self.assertIn("https://ai.studio/spend", str(ctx.exception))

    def test_ordinary_resource_exhausted_stays_rate_limit(self) -> None:
        with self.assertRaises(RateLimitError):
            _call(_client_error("Resource has been exhausted (e.g. check rate)."))


if __name__ == "__main__":
    unittest.main()


def test_declared_thinking_levels_reach_the_api_verbatim() -> None:
    # 2026-11-11: one shared map folded minimal → low and xhigh → high for every model.
    from llmx.providers import _google_thinking_level

    assert _google_thinking_level("gemini-3.5-flash-lite", "minimal") == "minimal"
    assert _google_thinking_level("gemini-4-carbon", "high") == "high"
    assert _google_thinking_level("gemini-4-argon", "high") == "high"
    # A model with no declared ladder keeps the old fold.
    assert _google_thinking_level("gemini-unregistered-x", "xhigh") == "high"


def test_carbon_offers_only_the_levels_that_think() -> None:
    # 2026-11-11 probe: the API accepts xhigh and max for Carbon but they think ~8x less
    # than high, so llmx must reject them instead of sending a weaker call.
    from llmx.providers import get_model_restriction

    assert get_model_restriction("gemini-4-carbon")["reasoning_effort_levels"] == [
        "low", "medium", "high",
    ]


def test_thinking_starved_google_reply_raises() -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    import pytest

    from llmx.providers import ModelError, _google_chat

    def reply(text, thoughts, visible):
        return SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason="FinishReason.MAX_TOKENS")],
            text=text,
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=visible,
                thoughts_token_count=thoughts, cached_content_token_count=None,
            ),
        )

    def call(response):
        client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock(return_value=response)))
        usage = {}
        with patch("llmx.providers.genai.Client", return_value=client), patch("llmx.usage_log.log_usage") as record:
            text = _google_chat("hi", "gemini-4-carbon", None, 1.0, 30, False, None, False, None, "high", usage_out=usage)
        return text, usage, record

    with pytest.raises(ModelError, match="no visible output"):
        call(reply("", 65533, 3))
    text, usage, record = call(reply("partial review", 60000, 5536))
    assert text == "partial review"
    assert "MAX_TOKENS" in usage["stop_reason"]
    assert "MAX_TOKENS" in record.call_args.kwargs["stop_reason"]
