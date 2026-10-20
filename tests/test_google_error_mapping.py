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
