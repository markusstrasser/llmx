"""Tests for mode surface."""

import unittest

from llmx.mode import resolve_mode


class TestResolveMode(unittest.TestCase):
    def test_default_chat(self):
        mode, source, lite, _ = resolve_mode(auth="subscription")
        self.assertEqual(mode, "chat")
        self.assertEqual(source, "default_policy")
        self.assertEqual(lite, "bare")

    def test_explicit_agent(self):
        mode, source, lite, _ = resolve_mode(mode="agent", auth="subscription")
        self.assertEqual(mode, "agent")
        self.assertEqual(source, "explicit")
        self.assertEqual(lite, "research")

    def test_lite_bare_deprecated(self):
        mode, source, lite, warns = resolve_mode(lite="bare", auth="subscription")
        self.assertEqual(mode, "chat")
        self.assertEqual(source, "lite_deprecated")
        self.assertTrue(warns)

    def test_api_forces_chat(self):
        mode, source, lite, warns = resolve_mode(mode="agent", auth="api")
        self.assertEqual(mode, "chat")
        self.assertEqual(source, "auth_forced")
        self.assertIsNone(lite)
        self.assertTrue(warns)

    def test_lite_mode_conflict(self):
        with self.assertRaises(ValueError):
            resolve_mode(mode="chat", lite="research", auth="subscription")


if __name__ == "__main__":
    unittest.main()
