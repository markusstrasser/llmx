import unittest

from llmx.providers import EXIT_QUOTA, EXIT_RATE_LIMIT, QuotaError, RateLimitError, _openai_rate_limit_error


class _SdkRateLimit(Exception):
    """Shape of the SDK's RateLimitError (checked against openai 3.19.0).

    `_make_status_error` passes `body.get("error", body)`, the inner error object,
    and APIError copies its `code` and `type` onto the exception.
    """

    def __init__(self, error: dict):
        super().__init__("Error code: 429")
        self.body = error
        self.code = error.get("code")
        self.type = error.get("type")


class TestOpenAIRateLimitMapping(unittest.TestCase):
    def test_no_credits_429_is_quota_not_transient(self):
        # 2026-09-29: a no-credit 429 exited 3 (retryable) because llmx looked for
        # body["error"]["code"] in a body the SDK had already unwrapped.
        exc = _SdkRateLimit(
            {
                "message": "You have no credits remaining.",
                "type": "insufficient_quota",
                "param": None,
                "code": "insufficient_quota",
            }
        )

        mapped = _openai_rate_limit_error(exc, "openai", "gpt-6-astra")

        self.assertIsInstance(mapped, QuotaError)
        self.assertEqual(mapped.exit_code, EXIT_QUOTA)

    def test_transient_429_stays_rate_limit(self):
        exc = _SdkRateLimit(
            {
                "message": "Rate limit reached for requests",
                "type": "requests",
                "param": None,
                "code": "rate_limit_exceeded",
            }
        )

        mapped = _openai_rate_limit_error(exc, "openai", "gpt-6-astra")

        self.assertIsInstance(mapped, RateLimitError)
        self.assertEqual(mapped.exit_code, EXIT_RATE_LIMIT)

    def test_wrapped_body_is_still_read(self):
        class Wrapped(Exception):
            body = {"error": {"type": "insufficient_quota", "code": None}}

        self.assertIsInstance(_openai_rate_limit_error(Wrapped(), "openai", "m"), QuotaError)


if __name__ == "__main__":
    unittest.main()
