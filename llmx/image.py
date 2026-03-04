"""Image generation using Gemini 3 Pro Image (Nano Banana Pro) API

Models:
- gemini-3-pro-image-preview: High-quality image generation with reasoning
- gemini-3-flash-preview: Fast image generation (fallback)

Reference: https://ai.google.dev/gemini-api/docs/image-generation
"""

import os
import sys
import base64
from pathlib import Path
from typing import Optional, Literal
from io import BytesIO

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


def check_genai_available():
    """Check if google-genai is installed"""
    try:
        from google import genai
        return True
    except ImportError:
        return False


def check_api_key() -> str:
    """Check for Gemini API key"""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.\n"
            "Get a key at: https://aistudio.google.com/apikey"
        )
    return key


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
                    # Auto-generate filename
                    base_name = prompt[:30].replace(" ", "_").replace("/", "-")
                    base_name = "".join(c for c in base_name if c.isalnum() or c in "_-")
                    saved_path = Path(f"{base_name}_{model}.png")

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
    return IMAGE_MODELS
