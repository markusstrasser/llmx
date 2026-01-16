"""
Vision analysis module for llmx.

Supports image and video analysis using Gemini 3 Flash/Pro.
"""

import mimetypes
from pathlib import Path
from typing import Optional

from .logger import logger

# Supported media types
IMAGE_MIMES = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.heic': 'image/heic',
    '.heif': 'image/heif',
}

VIDEO_MIMES = {
    '.mp4': 'video/mp4',
    '.mpeg': 'video/mpeg',
    '.mpg': 'video/mpg',
    '.mov': 'video/mov',
    '.avi': 'video/avi',
    '.webm': 'video/webm',
    '.wmv': 'video/wmv',
    '.flv': 'video/x-flv',
    '.3gp': 'video/3gpp',
}

# Size thresholds
INLINE_MAX_SIZE = 20 * 1024 * 1024  # 20MB for inline
VIDEO_INLINE_MAX = 100 * 1024 * 1024  # 100MB for video inline


def get_mime_type(file_path: Path) -> tuple[str, str]:
    """Get MIME type and media category for a file.

    Returns:
        Tuple of (mime_type, category) where category is 'image', 'video', or 'unknown'
    """
    suffix = file_path.suffix.lower()

    if suffix in IMAGE_MIMES:
        return IMAGE_MIMES[suffix], 'image'
    if suffix in VIDEO_MIMES:
        return VIDEO_MIMES[suffix], 'video'

    # Fallback to mimetypes
    mime, _ = mimetypes.guess_type(str(file_path))
    if mime:
        if mime.startswith('image/'):
            return mime, 'image'
        if mime.startswith('video/'):
            return mime, 'video'

    return 'application/octet-stream', 'unknown'


def analyze_media(
    file_paths: list[str],
    prompt: str,
    model: str = "flash",
    json_output: bool = False,
) -> str:
    """Analyze images or videos with Gemini.

    Args:
        file_paths: List of paths to image/video files
        prompt: Analysis prompt
        model: "flash" (fast) or "pro" (better quality)
        json_output: Return structured JSON

    Returns:
        Analysis text from the model
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai package required. Install with: pip install google-genai"
        )

    # Select model
    model_name = "gemini-3-flash-preview" if model == "flash" else "gemini-3-pro-preview"
    logger.info(f"Using model: {model_name}")

    client = genai.Client()
    contents = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        mime_type, category = get_mime_type(path)
        file_size = path.stat().st_size

        logger.info(f"Processing {category}: {path.name} ({file_size / 1024 / 1024:.1f}MB)")

        # Decide inline vs upload based on size and type
        use_upload = False
        if category == 'video' and file_size > VIDEO_INLINE_MAX:
            use_upload = True
        elif file_size > INLINE_MAX_SIZE:
            use_upload = True

        if use_upload:
            # Use Files API for large files
            logger.info(f"Uploading {path.name} via Files API...")
            uploaded = client.files.upload(file=str(path))
            contents.append(uploaded)
            logger.info(f"Upload complete: {uploaded.name}")
        else:
            # Inline for smaller files
            file_bytes = path.read_bytes()
            part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
            contents.append(part)

    # Add the prompt
    contents.append(prompt)

    # Generate response
    logger.info("Sending to Gemini...")

    config = None
    if json_output:
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )

    return response.text


def analyze_frames(
    frame_paths: list[str],
    prompt: str,
    model: str = "flash",
    sample_count: Optional[int] = None,
) -> str:
    """Analyze multiple frames (e.g., from gameplay recording).

    Args:
        frame_paths: List of frame image paths
        prompt: Analysis prompt
        model: "flash" or "pro"
        sample_count: If set, sample this many frames evenly from the list

    Returns:
        Analysis text
    """
    paths = sorted(frame_paths)

    # Sample frames if requested
    if sample_count and len(paths) > sample_count:
        step = len(paths) / sample_count
        paths = [paths[int(i * step)] for i in range(sample_count)]
        logger.info(f"Sampled {len(paths)} frames from {len(frame_paths)} total")

    return analyze_media(paths, prompt, model)
