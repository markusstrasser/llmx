"""Image generation helpers for llmx.

Backends:
- Gemini 3 Pro Image (Nano Banana Pro)
- OpenAI GPT Image 2

References:
- https://ai.google.dev/gemini-api/docs/image-generation
- https://platform.openai.com/docs/guides/image-generation
"""

import os
import sys
import base64
from pathlib import Path
from typing import Optional, Literal
from io import BytesIO
from urllib.request import urlopen

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
    "image2": "gpt-image-2",
    "gpt-image-2": "gpt-image-2",
    "gpt-image-2-2026-04-21": "gpt-image-2-2026-04-21",
}
OPENAI_IMAGE_SIZES = ["auto", "1024x1024", "1536x1024", "1024x1536"]
OPENAI_IMAGE_QUALITIES = ["auto", "low", "medium", "high"]
OPENAI_IMAGE_FORMATS = ["png", "jpeg", "webp"]


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
        with urlopen(image_url, timeout=60) as response:
            saved_path.write_bytes(response.read())
    else:
        raise RuntimeError("OpenAI image response did not include b64_json or url")

    logger.info(f"Image saved to {saved_path}")
    return saved_path


def generate_openai_image(
    prompt: str,
    output_path: Optional[str] = None,
    model: str = "gpt-image-2",
    input_images: Optional[list[str]] = None,
    size: str = "auto",
    quality: str = "auto",
    output_format: str = "png",
    input_fidelity: Optional[str] = None,
    n: int = 1,
) -> list[Path]:
    """
    Generate or edit images using OpenAI GPT Image models.

    Args:
        prompt: Text description of the image to generate or edit
        output_path: Path to save the image; for n>1, _1/_2 suffixes are added
        model: OpenAI image model, default gpt-image-2
        input_images: Optional source images for edit/reference workflows
        size: auto, 1024x1024, 1536x1024, or 1024x1536
        quality: auto, low, medium, or high
        output_format: png, jpeg, or webp
        input_fidelity: Optional high/low edit fidelity hint for models that accept it
        n: Number of output images

    Returns:
        Paths to saved images.
    """
    check_openai_api_key()

    if size not in OPENAI_IMAGE_SIZES:
        raise ValueError(f"Invalid OpenAI size: {size}. Valid: {OPENAI_IMAGE_SIZES}")
    if quality not in OPENAI_IMAGE_QUALITIES:
        raise ValueError(f"Invalid OpenAI quality: {quality}. Valid: {OPENAI_IMAGE_QUALITIES}")
    if output_format not in OPENAI_IMAGE_FORMATS:
        raise ValueError(f"Invalid OpenAI output format: {output_format}. Valid: {OPENAI_IMAGE_FORMATS}")
    if input_fidelity is not None and input_fidelity not in {"high", "low"}:
        raise ValueError("input_fidelity must be 'high' or 'low'")
    if n < 1:
        raise ValueError("n must be >= 1")

    from openai import OpenAI

    model_name = OPENAI_IMAGE_MODELS.get(model, model)
    client = OpenAI()
    files = []

    logger.info(f"Generating image with {model_name}", {
        "backend": "openai",
        "mode": "edit" if input_images else "generate",
        "size": size,
        "quality": quality,
        "format": output_format,
        "n": n,
        "prompt_length": len(prompt),
    })

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
            }
            if input_fidelity is not None:
                edit_kwargs["input_fidelity"] = input_fidelity
            response = client.images.edit(**edit_kwargs)
        else:
            response = client.images.generate(
                model=model_name,
                prompt=prompt,
                n=n,
                size=size,
                quality=quality,
                output_format=output_format,
            )
    finally:
        for f in files:
            f.close()

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
                model="gemini-3-flash-preview",
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
