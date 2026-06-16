"""Tests for auth surface."""

import unittest

from llmx.auth import auth_to_llmx_kwargs, resolve_auth


class TestResolveAuth(unittest.TestCase):
    def test_explicit_subscription(self):
        auth, source, lite, _ = resolve_auth(auth="subscription", provider="openai")
        self.assertEqual(auth, "subscription")
        self.assertEqual(source, "explicit")
        self.assertIsNone(lite)

    def test_explicit_api(self):
        auth, source, _, _ = resolve_auth(auth="api", provider="anthropic")
        self.assertEqual(auth, "api")
        self.assertEqual(source, "explicit")

    def test_anthropic_defaults_subscription(self):
        auth, source, lite, warns = resolve_auth(provider="anthropic")
        self.assertEqual(auth, "subscription")
        self.assertEqual(source, "default_policy")
        self.assertIsNone(lite)
        self.assertTrue(warns)

    def test_anthropic_direct_is_api(self):
        auth, source, _, _ = resolve_auth(provider="anthropic-direct")
        self.assertEqual(auth, "api")
        self.assertEqual(source, "explicit_provider")

    def test_api_only_deprecated_maps(self):
        auth, source, _, warns = resolve_auth(api_only=True, provider="google")
        self.assertEqual(auth, "api")
        self.assertEqual(source, "api_only_deprecated")
        self.assertTrue(warns)

    def test_both_auth_and_api_only_rejected(self):
        with self.assertRaises(ValueError):
            resolve_auth(auth="api", api_only=False, provider="google")

    def test_auth_to_llmx_kwargs(self):
        self.assertEqual(
            auth_to_llmx_kwargs("subscription"),
            {"api_only": False, "lite": "bare", "auth": "subscription", "mode": "chat"},
        )
        self.assertEqual(
            auth_to_llmx_kwargs("api"),
            {"api_only": True, "lite": None, "auth": "api", "mode": "chat"},
        )
        self.assertEqual(
            auth_to_llmx_kwargs("subscription", mode="agent"),
            {"api_only": False, "lite": "research", "auth": "subscription", "mode": "agent"},
        )


if __name__ == "__main__":
    unittest.main()
