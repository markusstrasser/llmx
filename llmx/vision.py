"""
Vision analysis for llmx — provider-agnostic.

This module owns MEDIA semantics (MIME typing, size thresholds, provider-native
encoding). It does NOT own dispatch: `analyze_media` routes through
`providers.chat(..., media=[...])` like every other call.

That split is the whole point of the 2026-07-22 rewrite. The previous version
built its own `genai.Client()` and called Gemini directly, which meant it:
  - was hardcoded to Gemini (no GPT, no Claude, no choice of model);
  - bypassed the spend guard AND the Gemini critique-only policy gate;
  - never wrote to ~/.claude/llmx-usage.jsonl, so callers had no token counts —
    evals/figure_vision_bakeoff had to substitute a `len(response)/4` proxy and
    label its costs order-of-magnitude estimates;
  - broke outright once the Gemini key was scoped to GEMINI_API_KEY_CRITIQUE_ONLY,
    because a bare genai.Client() finds no key.
Routing through the one dispatch path fixes all four at once.
"""

import base64
import contextlib
import io
import mimetypes
from pathlib import Path
from typing import Optional

from .logger import logger

# Supported media types
IMAGE_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}

VIDEO_MIMES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".flv": "video/x-flv",
    ".3gp": "video/3gpp",
}

# Size thresholds
INLINE_MAX_SIZE = 20 * 1024 * 1024  # 20MB for inline
VIDEO_INLINE_MAX = 100 * 1024 * 1024  # 100MB for video inline

DEFAULT_VISION_MODEL = "gemini-3.8-flash"


class UnsupportedMediaError(Exception):
    """Raised when a provider cannot carry the media it was handed."""


def get_mime_type(file_path: Path) -> tuple[str, str]:
    """Get MIME type and media category for a file.

    Returns:
        Tuple of (mime_type, category) where category is 'image', 'video', or 'unknown'
    """
    suffix = file_path.suffix.lower()

    if suffix in IMAGE_MIMES:
        return IMAGE_MIMES[suffix], "image"
    if suffix in VIDEO_MIMES:
        return VIDEO_MIMES[suffix], "video"

    # Fallback to mimetypes
    mime, _ = mimetypes.guess_type(str(file_path))
    if mime:
        if mime.startswith("image/"):
            return mime, "image"
        if mime.startswith("video/"):
            return mime, "video"

    return "application/octet-stream", "unknown"


def resolve_media(media: list) -> list[tuple[Path, str, str, int]]:
    """Validate paths and return (path, mime, category, size) for each.

    Fails loud on a missing file — a vision call that silently analyses fewer
    images than it was given produces a confident answer about the wrong input.
    """
    out = []
    for item in media:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {item}")
        mime, category = get_mime_type(path)
        size = path.stat().st_size
        logger.info(f"Processing {category}: {path.name} ({size / 1024 / 1024:.1f}MB)")
        out.append((path, mime, category, size))
    return out


def build_google_contents(media: list, prompt: str, client) -> list:
    """Native Gemini `contents`: Parts for each file, then the prompt text.

    Large files go through the Files API on the caller's client (which already
    holds the promoted, policy-gated key).
    """
    from google.genai import types

    contents = []
    for path, mime, category, size in resolve_media(media):
        too_big = size > (VIDEO_INLINE_MAX if category == "video" else INLINE_MAX_SIZE)
        if too_big:
            logger.info(f"Uploading {path.name} via Files API...")
            contents.append(client.files.upload(file=str(path)))
        else:
            contents.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
    contents.append(prompt)
    return contents


def build_openai_content(media: list, prompt: str) -> list:
    """OpenAI-compatible multimodal `content`: text part + base64 image parts."""
    content: list = [{"type": "text", "text": prompt}]
    for path, mime, category, size in resolve_media(media):
        if category != "image":
            raise UnsupportedMediaError(
                f"{path.name} is {category}; OpenAI-compatible endpoints accept images only. "
                f"Route video to Gemini (-p google)."
            )
        if size > INLINE_MAX_SIZE:
            raise UnsupportedMediaError(
                f"{path.name} is {size / 1024 / 1024:.1f}MB, over the {INLINE_MAX_SIZE // 1024 // 1024}MB "
                f"inline limit for this provider. Route to Gemini (-p google), which can upload it."
            )
        b64 = base64.b64encode(path.read_bytes()).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    return content


def analyze_media(
    file_paths: list[str],
    prompt: str,
    model: str = DEFAULT_VISION_MODEL,
    json_output: bool = False,
    provider: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """Analyze images or videos with any vision-capable model.

    `model` is a real model id (`gemini-3.6-flash`, `gpt-5.6-sol`, ...). Provider
    is inferred from it unless given explicitly.
    """
    from .providers import chat, infer_provider_from_model

    resolved_provider = provider or infer_provider_from_model(model) or "google"
    logger.info(f"Using model: {model} (provider: {resolved_provider})")

    # The provider functions print to stdout as a side effect (that is how
    # `llmx chat` emits). analyze_media's contract is to RETURN the text, so
    # swallow that print here — otherwise every caller double-emits.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        text = chat(
            prompt=prompt,
            provider=resolved_provider,
            model=model,
            temperature=1.0,
            reasoning_effort=reasoning_effort,
            stream=False,
            debug=False,
            json_output=json_output,
            timeout=timeout,
            media=list(file_paths),
        )
    text = text or buf.getvalue().strip()
    # chat() is Optional[str]; an empty vision result is a failure, not "".
    if not text:
        raise RuntimeError(f"{model} returned no content for {len(file_paths)} media file(s)")
    return text


def analyze_frames(
    frame_paths: list[str],
    prompt: str,
    model: str = DEFAULT_VISION_MODEL,
    sample_count: Optional[int] = None,
    **kwargs,
) -> str:
    """Analyze multiple frames (e.g., from gameplay recording).

    `sample_count` evenly samples that many frames from the list.
    """
    paths = sorted(frame_paths)

    if sample_count and len(paths) > sample_count:
        step = len(paths) / sample_count
        paths = [paths[int(i * step)] for i in range(sample_count)]
        logger.info(f"Sampled {len(paths)} frames from {len(frame_paths)} total")

    return analyze_media(paths, prompt, model, **kwargs)
