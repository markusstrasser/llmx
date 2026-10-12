"""GPT vision defaults to the codex-cli subscription route (`codex exec -i`).

2026-10-12: `llmx vision img.jpg -m gpt-6-luna` failed with "Invalid API key for
openai" because vision dispatched codex-served GPT models to the metered API.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llmx.vision import analyze_media


def _boom(*args, **kwargs):
    raise AssertionError("OpenAI API client must not be constructed on the subscription route")


class TestVisionCodexRoute(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.a = Path(self.tmp.name) / "a.jpg"
        self.b = Path(self.tmp.name) / "b.png"
        self.v = Path(self.tmp.name) / "c.mp4"
        for p in (self.a, self.b, self.v):
            p.write_bytes(b"\x00")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_images_attach_via_codex_dash_i(self) -> None:
        with patch("llmx.cli_backends.subprocess.Popen") as popen, patch(
            "llmx.providers.OpenAI", side_effect=_boom
        ), patch("llmx.cli_backends.shutil.which", return_value="/usr/bin/codex"):
            proc = popen.return_value
            proc.pid = 123
            proc.returncode = 0
            proc.communicate.return_value = ("A test image.", "")
            out = analyze_media([str(self.a), str(self.b)], "what?", "gpt-6-luna", reasoning_effort="low")
        self.assertEqual(out, "A test image.")
        argv = next(c.args[0] for c in popen.call_args_list if c.args and c.args[0][0] == "codex")
        pairs = list(zip(argv, argv[1:]))
        self.assertIn(("-i", str(self.a.resolve())), pairs)
        self.assertIn(("-i", str(self.b.resolve())), pairs)
        self.assertIn(("-m", "gpt-6-luna"), pairs)
        self.assertIn(("-s", "read-only"), pairs)
        self.assertIn(("-c", 'model_reasoning_effort="low"'), pairs)
        self.assertEqual(argv[-1], "what?")

    def test_video_fails_loud_without_api(self) -> None:
        with patch("llmx.cli_backends.subprocess.Popen") as popen, patch(
            "llmx.providers.OpenAI", side_effect=_boom
        ), patch("llmx.cli_backends.shutil.which", return_value="/usr/bin/codex"):
            with self.assertRaises(Exception) as ctx:
                analyze_media([str(self.v)], "what?", "gpt-6-luna")
        self.assertIn("images only", str(ctx.exception))
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
