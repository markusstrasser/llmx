"""Image generation helpers for llmx.

Backends:
- Gemini 3 Pro Image (Nano Banana Pro)
- OpenAI GPT Image 2.6 (Sunburst, Flare)

References:
- https://ai.google.dev/gemini-api/docs/image-generation
- https://platform.openai.com/docs/guides/image-generation
"""

import os
import sys
import base64
from pathlib import Path
from typing import Any, Optional, Literal
from io import BytesIO

from . import keepalive
from .logger import logger

# Model configurations (Jan 2026)
# See: https://ai.google.dev/gemini-api/docs/image-generation
# NOTE: Only dedicated image models support image generation output.
# There is NO Gemini 3 Flash Image model - only Pro exists in the 3 series.
IMAGE_MODELS = {
    "flash": {
        # No Gemini 3 Flash Image exists, use Pro for all Gemini 3 image generation
        "name": "gemini-3-pro-image-preview",
        "description": "Gemini 3 Pro Image (Nano Banana Pro)",
        "max_resolution": "4K",
        "supports_image_config": True,
    },
    "pro": {
        "name": "gemini-3-pro-image-preview",  # Nano Banana Pro
        "description": "Gemini 3 Pro Image (Nano Banana Pro, supports 4K)",
        "max_resolution": "4K",
        "supports_image_config": True,
    },
}

# Valid aspect ratios per Gemini API docs
ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
# Resolutions must be uppercase K
RESOLUTIONS = ["1K", "2K", "4K"]
OPENAI_IMAGE_MODELS = {
    "sunburst": "gpt-image-2.6-sunburst",
    "flare": "gpt-image-2.6-flare",
    "gpt-image-2.6": "gpt-image-2.6-sunburst",
    "gpt-image-2.5": "gpt-image-2.5-sunburst",
    "image2": "gpt-image-2",
    "gpt-image-2": "gpt-image-2",
    "gpt-image-2-2026-04-21": "gpt-image-2-2026-04-21",
}
OPENAI_IMAGE_DEFAULT = "gpt-image-2.6-sunburst"  # Images 2.6, 2026-11-04
OPENAI_IMAGE_QUALITIES = ["auto", "low", "medium", "high", "xhigh", "max"]
OPENAI_IMAGE_FORMATS = ["png", "jpeg", "webp", "svg", "gif"]
OPENAI_IMAGE_COMPLEXITIES = ["low", "medium", "high"]  # API default when omitted: high
# The image model draws a reference; this text model writes the SVG from it. The API
# requires svg.model on accounts with SVG model selection (live 400, 2026-11-05).
OPENAI_SVG_AUTHOR_DEFAULT = "gpt-6.1-sol"


def _validate_openai_request(
    output_path: Optional[str],
    input_images: Optional[list[str]],
    size: str,
    quality: str,
    output_format: str,
    input_fidelity: Optional[str],
    n: int,
    complexity: Optional[str],
    frame_count: Optional[int],
    frame_duration_ms: Optional[int],
    markup: Optional[str],
    svg_model: Optional[str],
    svg_effort: Optional[str],
) -> None:
    """Enforce the request shapes the image guide documents, independent of model.

    Which model accepts which feature stays the API's call: a rejection is free and
    a local model table would need editing at every release.
    """
    if size != "auto":
        width, _, height = size.partition("x")
        if not (width.isdigit() and height.isdigit() and int(width) % 16 == 0 and int(height) % 16 == 0
                and int(width) > 0 and int(height) > 0):
            raise ValueError(f"Invalid OpenAI size: {size}. Use auto or WIDTHxHEIGHT in multiples of 16")
    if quality not in OPENAI_IMAGE_QUALITIES:
        raise ValueError(f"Invalid OpenAI quality: {quality}. Valid: {OPENAI_IMAGE_QUALITIES}")
    if output_format not in OPENAI_IMAGE_FORMATS:
        raise ValueError(f"Invalid OpenAI output format: {output_format}. Valid: {OPENAI_IMAGE_FORMATS}")
    if input_fidelity is not None and input_fidelity not in {"high", "low"}:
        raise ValueError("input_fidelity must be 'high' or 'low'")
    if n < 1:
        raise ValueError("n must be >= 1")
    if complexity is not None and complexity not in OPENAI_IMAGE_COMPLEXITIES:
        raise ValueError(f"Invalid complexity: {complexity}. Valid: {OPENAI_IMAGE_COMPLEXITIES}")
    if output_format in {"svg", "gif"}:
        if input_images:
            raise ValueError(f"{output_format} output is generation-only; the edit endpoint rejects it")
        if output_path and Path(output_path).suffix.lower() != f".{output_format}":
            raise ValueError(f"{output_format} output needs a .{output_format} path, got {output_path}")
    if output_format == "svg" and n != 1:
        raise ValueError("svg output requires n=1")
    if output_format != "svg" and (svg_model is not None or svg_effort is not None):
        raise ValueError("svg_model and svg_effort apply only to svg output")
    if output_format != "gif" and (frame_count is not None or frame_duration_ms is not None):
        raise ValueError("frame_count and frame_duration_ms apply only to gif output")
    if frame_count is not None and not 2 <= frame_count <= 16:  # live 400 above 16; the guide says 64
        raise ValueError("frame_count must be between 2 and 16")
    if frame_duration_ms is not None and not 20 <= frame_duration_ms <= 2000:
        raise ValueError("frame_duration_ms must be between 20 and 2000")
    if markup is not None:
        if not input_images:
            raise ValueError("markup annotates an edit: pass the image it was drawn on as the first input image")
        if not Path(markup).is_file():
            raise FileNotFoundError(f"Markup image not found: {markup}")


def check_genai_available():
    """Check if google-genai is installed"""
    try:
        from google import genai
        return True
    except ImportError:
        return False


def check_api_key() -> str:
    """Check for Gemini API key (env vars + macOS Keychain)."""
    from .providers import check_api_key as _provider_check, _get_api_key
    _provider_check("google")  # raises RuntimeError with hint if missing
    key = _get_api_key("google")
    if not key:  # defense in depth — _provider_check should have raised
        raise RuntimeError("API key not found for google after Keychain fallback")
    return key


def check_openai_api_key() -> str:
    """Check for OpenAI API key (env vars + macOS Keychain)."""
    from .providers import check_api_key as _provider_check, _get_api_key
    _provider_check("openai")
    key = _get_api_key("openai")
    if not key:
        raise RuntimeError("API key not found for openai after Keychain fallback")
    return key


def _safe_stem(prompt: str) -> str:
    base_name = prompt[:30].replace(" ", "_").replace("/", "-")
    return "".join(c for c in base_name if c.isalnum() or c in "_-") or "llmx_image"


def _output_path_for_index(output_path: Optional[str], prompt: str, index: int, total: int, suffix: str) -> Path:
    if output_path:
        path = Path(output_path)
        if total == 1:
            return path
        return path.with_name(f"{path.stem}_{index + 1}{path.suffix or suffix}")
    return Path(f"{_safe_stem(prompt)}_{index + 1}{suffix}")


def _write_openai_image_item(item, output_path: Optional[str], prompt: str, index: int, total: int, output_format: str) -> Path:
    suffix = f".{'jpg' if output_format == 'jpeg' else output_format}"
    saved_path = _output_path_for_index(output_path, prompt, index, total, suffix)
    saved_path.parent.mkdir(parents=True, exist_ok=True)

    image_b64 = getattr(item, "b64_json", None)
    image_url = getattr(item, "url", None)
    if image_b64:
        saved_path.write_bytes(base64.b64decode(image_b64))
    elif image_url:
        with keepalive.open_url(image_url, timeout=60) as response:
            saved_path.write_bytes(response.read())
    else:
        raise RuntimeError("OpenAI image response did not include b64_json or url")

    logger.info(f"Image saved to {saved_path}")
    return saved_path


CODEX_GENERATED_DIR = Path.home() / ".codex" / "generated_images"


def _log_codex_image_usage(proc, before, started: float) -> None:
    """Log the codex call's tokens (image_gen bills to the plan; tokens still cost latency)."""
    import time

    from .cli_backends import _codex_session_id, _latest_codex_rollout_usage
    from .usage_log import log_usage

    usage, note = _latest_codex_rollout_usage(
        before, started_at=started, session_id=_codex_session_id(proc.stderr or "")
    )
    log_usage(
        provider="openai", model="codex-image_gen", transport="codex-cli", reasoning_effort=None,
        prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=usage.get("reasoning_tokens"), cached_tokens=usage.get("cached_tokens"),
        latency_s=time.time() - started, source="llmx-image", note=note, billing="subscription",
        error=None if proc.returncode == 0 else f"rc={proc.returncode}",
    )


def _log_api_image_usage(response, model_name: str, quality: str, complexity: Optional[str], started: float) -> None:
    """Log the Images API call's tokens, split by modality because the rates differ.

    SVG output bills the authoring model too, reported apart from the image usage in
    ``svg_usage`` (image generation guide, "GPT Image 2.6 costs"), so it gets its own row.
    """
    import time

    from .usage_log import log_usage

    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    log_usage(
        provider="openai", model=model_name, transport="images-api", reasoning_effort=quality,
        prompt_tokens=getattr(usage, "input_tokens", None),
        completion_tokens=getattr(usage, "output_tokens", None), reasoning_tokens=None,
        cached_tokens=getattr(details, "cached_tokens", None), latency_s=time.time() - started,
        source="llmx-image", billing="metered",
        image_input_tokens=getattr(details, "image_tokens", None),
        text_output_tokens=getattr(output_details, "text_tokens", None),
        note="; ".join(filter(None, [
            f"complexity={complexity}" if complexity else None,
            None if usage else "response carried no usage",
        ])) or None,
    )
    svg = getattr(response, "svg_usage", None)
    if svg:
        log_usage(
            provider="openai", model=svg.model, transport="images-api", reasoning_effort=None,
            prompt_tokens=svg.input_tokens, completion_tokens=svg.output_tokens,
            reasoning_tokens=svg.reasoning_tokens, cached_tokens=None, latency_s=time.time() - started,
            source="llmx-image", billing="metered", note=f"svg author for {model_name}",
            # The SDK does not say whether output_tokens holds the reasoning; the total does.
            completion_includes_reasoning=svg.total_tokens == svg.input_tokens + svg.output_tokens,
        )


def generate_codex_image(
    prompt: str,
    output_path: Optional[str] = None,
    input_images: Optional[list[str]] = None,
    timeout: int = 900,
) -> list[Path]:
    """Generate one image through codex-cli's built-in image_gen tool ($0, ChatGPT plan).

    The codex tool does not expose its model id; it serves the plan's current GPT
    Image model. codex is asked to copy the file to ``output_path``; if it doesn't,
    the newest file under ~/.codex/generated_images written after launch is used.
    """
    import shutil
    import subprocess
    import time

    if not shutil.which("codex"):
        raise RuntimeError("codex CLI not found; install it or use --auth api")
    out = Path(output_path or f"{_safe_stem(prompt)}.png").expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    refs = [str(Path(p).expanduser().resolve()) for p in (input_images or [])]
    for r in refs:
        if not Path(r).is_file():
            raise FileNotFoundError(f"Input image not found: {r}")
    ref_note = (" Use the attached image(s) as visual reference." if refs else "")
    instruction = (
        "Use your image generation tool to create exactly one image from the prompt below."
        f"{ref_note} Do not alter the prompt's content. Copy the generated file to {out} "
        "(convert to PNG if needed) and reply with only that path.\n\nPROMPT:\n" + prompt
    )
    cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "workspace-write", "-C", str(out.parent)]
    for r in refs:
        cmd += ["-i", r]
    cmd += ["--", instruction]  # `-i` is multi-value in codex's parser; `--` ends it
    from .cli_backends import _codex_rollout_snapshot

    before = _codex_rollout_snapshot()
    started = time.time()
    logger.info("Generating image via codex image_gen (subscription)", {"refs": len(refs), "prompt_length": len(prompt)})
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    _log_codex_image_usage(proc, before, started)
    if out.is_file() and out.stat().st_mtime >= started:
        return [out]
    fresh = sorted(
        (p for p in CODEX_GENERATED_DIR.rglob("*") if p.is_file() and p.stat().st_mtime >= started),
        key=lambda p: p.stat().st_mtime,
    ) if CODEX_GENERATED_DIR.is_dir() else []
    if fresh:
        shutil.copyfile(fresh[-1], out)
        return [out]
    tail = (proc.stdout + proc.stderr)[-1500:]
    raise RuntimeError(f"codex produced no image (rc={proc.returncode}). Output tail:\n{tail}")


def generate_openai_image(
    prompt: str,
    output_path: Optional[str] = None,
    model: str = OPENAI_IMAGE_DEFAULT,
    input_images: Optional[list[str]] = None,
    size: str = "auto",
    quality: str = "auto",
    output_format: str = "png",
    input_fidelity: Optional[str] = None,
    n: int = 1,
    complexity: Optional[str] = None,
    frame_count: Optional[int] = None,
    frame_duration_ms: Optional[int] = None,
    markup: Optional[str] = None,
    svg_model: Optional[str] = None,
    svg_effort: Optional[str] = None,
) -> list[Path]:
    """
    Generate or edit images using OpenAI GPT Image models.

    Args:
        prompt: Text description of the image to generate or edit
        output_path: Path to save the image; for n>1, _1/_2 suffixes are added
        model: OpenAI image model, default gpt-image-2.6-sunburst
        input_images: Optional source images for edit/reference workflows
        size: auto or WIDTHxHEIGHT in multiples of 16
        quality: auto, low, medium, high, xhigh, or max
        output_format: png, jpeg, webp, svg, or gif (svg/gif: generation only)
        input_fidelity: Optional high/low edit fidelity hint for models that accept it
        n: Number of output images
        complexity: low, medium, or high planning effort; omitted means the API's high
        frame_count: GIF frames, 2-16 (API default 9)
        frame_duration_ms: GIF frame duration, 20-2000 ms (API default 200)
        markup: Transparent PNG annotation layer matching the first input image's size
        svg_model: Text model that authors SVG output, default gpt-6.1-sol
        svg_effort: Reasoning effort for the SVG author; omitted means its default

    Returns:
        Paths to saved images.
    """
    _validate_openai_request(
        output_path, input_images, size, quality, output_format, input_fidelity, n,
        complexity, frame_count, frame_duration_ms, markup, svg_model, svg_effort,
    )
    check_openai_api_key()

    import time

    from openai import OpenAI

    started = time.time()
    model_name = OPENAI_IMAGE_MODELS.get(model, model)
    from .spend_guard import enforce_daily_cap

    enforce_daily_cap(model_name)
    client = OpenAI()
    files = []

    logger.info(f"Generating image with {model_name}", {
        "backend": "openai",
        "mode": "edit" if input_images else "generate",
        "size": size,
        "quality": quality,
        "format": output_format,
        "complexity": complexity,
        "n": n,
        "prompt_length": len(prompt),
    })

    # Sent only when chosen: older models and output formats reject the fields.
    optional: dict[str, Any] = {"complexity": complexity} if complexity is not None else {}
    try:
        if input_images:
            for image_path in input_images:
                path = Path(image_path)
                if not path.is_file():
                    raise FileNotFoundError(f"Input image not found: {image_path}")
                files.append(path.open("rb"))

            edit_kwargs = {
                "model": model_name,
                "image": files if len(files) > 1 else files[0],
                "prompt": prompt,
                "n": n,
                "size": size,
                "quality": quality,
                "output_format": output_format,
                **optional,
            }
            if input_fidelity is not None:
                edit_kwargs["input_fidelity"] = input_fidelity
            if markup is not None:
                markup_file = Path(markup).open("rb")
                files.append(markup_file)
                edit_kwargs["markup"] = (Path(markup).name, markup_file, "image/png")
            response = client.images.edit(**edit_kwargs)
        else:
            if frame_count is not None:
                optional["frame_count"] = frame_count
            if frame_duration_ms is not None:
                optional["frame_duration_ms"] = frame_duration_ms
            if output_format == "svg":
                svg: dict[str, Any] = {"model": svg_model or OPENAI_SVG_AUTHOR_DEFAULT}
                if svg_effort is not None:
                    svg["reasoning"] = {"effort": svg_effort}
                optional["svg"] = svg
            response = client.images.generate(
                model=model_name,
                prompt=prompt,
                n=n,
                size=size,
                quality=quality,
                output_format=output_format,
                **optional,
            )
    finally:
        for f in files:
            f.close()

    _log_api_image_usage(response, model_name, quality, complexity, started)
    data = getattr(response, "data", None) or []
    if not data:
        logger.warn("No image was generated in the response")
        return []

    return [
        _write_openai_image_item(item, output_path, prompt, index, len(data), output_format)
        for index, item in enumerate(data)
    ]


def generate_image(
    prompt: str,
    output_path: Optional[str] = None,
    model: Literal["flash", "pro"] = "flash",
    aspect_ratio: str = "1:1",
    resolution: str = "1K",
    show_text: bool = True,
) -> Optional[Path]:
    """
    Generate an image using Gemini Nano Banana models.

    Args:
        prompt: Text description of the image to generate
        output_path: Path to save the image (default: auto-generated)
        model: "flash" (fast) or "pro" (high quality)
        aspect_ratio: Aspect ratio like "1:1", "16:9", "4:3"
        resolution: "1K", "2K", or "4K" (pro model only for 2K/4K)
        show_text: Whether to print any text response from the model

    Returns:
        Path to the saved image, or None if generation failed
    """
    if not check_genai_available():
        raise RuntimeError(
            "google-genai package not installed.\n"
            "Install with: uv pip install google-genai pillow"
        )

    from google import genai
    from google.genai import types

    api_key = check_api_key()

    # Validate inputs
    if aspect_ratio not in ASPECT_RATIOS:
        raise ValueError(f"Invalid aspect ratio: {aspect_ratio}. Valid: {ASPECT_RATIOS}")

    if resolution not in RESOLUTIONS:
        raise ValueError(f"Invalid resolution: {resolution}. Valid: {RESOLUTIONS}")

    if resolution in ["2K", "4K"] and model != "pro":
        logger.warn(f"Resolution {resolution} requires 'pro' model, switching from 'flash'")
        model = "pro"

    model_config = IMAGE_MODELS[model]
    model_name = model_config["name"]

    logger.info(f"Generating image with {model_name}", {
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "prompt_length": len(prompt),
    })

    # Create client
    client = genai.Client(api_key=api_key)

    # Configure image generation
    # Note: image_size must be uppercase (1K, 2K, 4K)
    # Note: Only gemini-3-pro-image-preview supports image_config with aspect_ratio/image_size
    supports_image_config = model_config.get("supports_image_config", False)

    if supports_image_config:
        config = types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution,  # Must be uppercase: "1K", "2K", "4K"
            ),
        )
    else:
        # Flash/native models - simpler config
        config = types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
        )

    # Generate
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt],
            config=config,
        )
    except Exception as e:
        error_msg = str(e)
        if "safety" in error_msg.lower():
            raise RuntimeError(f"Image generation blocked by safety filter. Try a different prompt.")
        # Try fallback to flash model if pro fails
        if model == "pro" and "not found" in error_msg.lower():
            logger.warn(f"Model {model_name} not available, trying flash model")
            return generate_image(prompt, output_path, "flash", aspect_ratio, "1K", show_text)
        raise RuntimeError(f"Image generation failed: {error_msg}")

    # Process response
    image_saved = False
    saved_path = None

    for part in response.parts:
        if hasattr(part, 'text') and part.text and show_text:
            print(part.text)

        if hasattr(part, 'inline_data') and part.inline_data:
            # Got image data
            try:
                from PIL import Image

                # Decode image
                image_data = part.inline_data.data
                if isinstance(image_data, str):
                    image_data = base64.b64decode(image_data)

                image = Image.open(BytesIO(image_data))

                # Determine output path
                if output_path:
                    saved_path = Path(output_path)
                else:
                    saved_path = Path(f"{_safe_stem(prompt)}_{model}.png")

                # Ensure parent directory exists
                saved_path.parent.mkdir(parents=True, exist_ok=True)

                # Save image
                image.save(saved_path)
                image_saved = True

                logger.info(f"Image saved to {saved_path}", {
                    "size": f"{image.width}x{image.height}",
                    "format": image.format or "PNG",
                })

            except ImportError:
                raise RuntimeError(
                    "PIL/Pillow not installed. Install with: uv pip install pillow"
                )

    if not image_saved:
        logger.warn("No image was generated in the response")
        return None

    return saved_path


def generate_svg(
    prompt: str,
    output_path: Optional[str] = None,
    model: Literal["flash", "pro"] = "pro",
) -> Optional[Path]:
    """
    Generate an SVG using Gemini by asking for SVG code.

    Args:
        prompt: Description of the SVG to generate
        output_path: Path to save the SVG (default: auto-generated)
        model: "flash" or "pro" (pro recommended for SVG)

    Returns:
        Path to the saved SVG, or None if generation failed
    """
    if not check_genai_available():
        raise RuntimeError(
            "google-genai package not installed.\n"
            "Install with: uv pip install google-genai"
        )

    from google import genai
    from google.genai import types

    api_key = check_api_key()

    model_config = IMAGE_MODELS[model]
    model_name = model_config["name"]

    # Craft SVG-specific prompt
    svg_prompt = f"""Generate clean, well-structured SVG code for the following:

{prompt}

Requirements:
- Output ONLY the SVG code, no explanations
- Use a viewBox for scalability
- Keep the SVG simple and efficient
- Use appropriate colors and shapes
- Make it suitable for use as a game asset or icon

Output the raw SVG code starting with <svg and ending with </svg>."""

    logger.info(f"Generating SVG with {model_name}", {
        "prompt_length": len(prompt),
    })

    # Create client - for SVG we just want text output
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model_name.replace("-image", ""),  # Use non-image variant for text
            contents=[svg_prompt],
        )
    except Exception as e:
        # Fallback to text model if image model doesn't work for text
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[svg_prompt],
            )
        except Exception as e2:
            raise RuntimeError(f"SVG generation failed: {e2}")

    # Extract SVG from response
    text = response.text if hasattr(response, 'text') else str(response)

    # Find SVG content
    svg_start = text.find("<svg")
    svg_end = text.rfind("</svg>")

    if svg_start == -1 or svg_end == -1:
        logger.warn("No valid SVG found in response")
        print("Response:", text[:500])
        return None

    svg_content = text[svg_start:svg_end + 6]

    # Determine output path
    if output_path:
        saved_path = Path(output_path)
    else:
        base_name = prompt[:30].replace(" ", "_").replace("/", "-")
        base_name = "".join(c for c in base_name if c.isalnum() or c in "_-")
        saved_path = Path(f"{base_name}.svg")

    # Save SVG
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_text(svg_content)

    logger.info(f"SVG saved to {saved_path}", {
        "size_bytes": len(svg_content),
    })

    return saved_path


def list_models():
    """List available image generation models"""
    return {
        "google": IMAGE_MODELS,
        "openai": OPENAI_IMAGE_MODELS,
    }
