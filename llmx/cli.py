#!/usr/bin/env python3
"""
llmx - Unified LLM CLI via native SDKs

Usage:
    llmx "your prompt"
    llmx --model gpt-5.5 "your prompt"
    cat file.txt | llmx --model claude-sonnet-4-6 "review"
    llmx --compare "question"
    llmx image "a cute robot" -o robot.png
    llmx svg "physics momentum arrow icon" -o momentum.svg
"""

import sys
import os
import json
from pathlib import Path
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

# Load .env from current directory and parents
load_dotenv()
load_dotenv(Path.cwd().parent / ".env")
load_dotenv(Path.cwd().parent.parent / ".env")

from . import __version__
from .providers import (
    chat, compare as compare_providers, list_providers, infer_provider_from_model,
    LlmxError, RateLimitError, QuotaError, TimeoutError_, SearchUnavailableError,
    EXIT_GENERAL, PROVIDER_CONFIGS, get_model_name, get_model_restriction,
)
from .cli_backends import preferred_cli_provider, needs_api_fallback, CLI_PROVIDERS
from .logger import configure_logger, logger

console = Console()


class _TeeWriter:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, stdout, file):
        self._stdout = stdout
        self._file = file

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._stdout, name)


# Subcommand names for detection
SUBCOMMANDS = {"image", "svg", "vision", "research", "batch"}


# ============================================================================
# Image Generation Commands
# ============================================================================

@click.command("image")
@click.argument("prompt", nargs=-1, required=True)
@click.option("-o", "--output", help="Output file path (default: auto-generated)")
@click.option(
    "-p", "--provider",
    type=click.Choice(["openai", "google"]),
    default="openai",
    help="Image backend: openai uses GPT Image 2; google uses Gemini 3 Pro Image",
)
@click.option(
    "-m", "--model",
    default=None,
    help="Model alias/name. OpenAI default: gpt-image-2. Google: flash/pro",
)
@click.option(
    "--aspect-ratio", "-a",
    default="1:1",
    help="Google aspect ratio: 1:1, 16:9, 9:16, 4:3, 3:4, 5:4, 4:5",
)
@click.option(
    "--resolution", "-r",
    type=click.Choice(["1K", "2K", "4K"]),
    default="1K",
    help="Google resolution: 1K, 2K, 4K",
)
@click.option(
    "--input-image", "-i",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="OpenAI edit/reference image. Repeat for multiple reference images.",
)
@click.option(
    "--size",
    default="auto",
    help="OpenAI output size: auto, 1024x1024, 1536x1024, 1024x1536",
)
@click.option(
    "--quality",
    default="auto",
    help="OpenAI quality: auto, low, medium, high",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["png", "jpeg", "webp"]),
    default="png",
    help="OpenAI output format",
)
@click.option(
    "--input-fidelity",
    type=click.Choice(["high", "low"]),
    default=None,
    help="Optional OpenAI edit/reference input fidelity for models that accept it",
)
@click.option(
    "-n", "--count",
    type=click.IntRange(1, 10),
    default=1,
    help="Number of OpenAI output images",
)
@click.option("--debug", is_flag=True, help="Debug logging")
def image_cmd(prompt, output, provider, model, aspect_ratio, resolution, input_image, size, quality, output_format, input_fidelity, count, debug):
    """Generate or edit images.

    Examples:
        llmx image "a cute robot mascot" -o robot.png
        llmx image -i photobooth.jpg -o beard.png "same person, short crop, trimmed beard"
        llmx image --provider google "game background forest" -m pro -r 2K
    """
    configure_logger(debug=debug)

    prompt_text = " ".join(prompt)

    try:
        if provider == "openai":
            from .image import generate_openai_image

            results = generate_openai_image(
                prompt=prompt_text,
                output_path=output,
                model=model or "gpt-image-2",
                input_images=list(input_image) or None,
                size=size,
                quality=quality,
                output_format=output_format,
                input_fidelity=input_fidelity,
                n=count,
            )
            if results:
                for result in results:
                    click.echo(f"Image saved: {result}")
            else:
                click.echo("No image was generated.", err=True)
                sys.exit(1)
            return

        from .image import generate_image
        google_model = model or "pro"
        if google_model not in {"flash", "pro"}:
            raise click.ClickException("Google image backend model must be 'flash' or 'pro'")
        if input_image:
            raise click.ClickException("--input-image is currently supported only with --provider openai")
        if count != 1:
            raise click.ClickException("--count is currently supported only with --provider openai")
        result = generate_image(
            prompt=prompt_text,
            output_path=output,
            model=google_model,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
        )
        if result:
            click.echo(f"Image saved: {result}")
        else:
            click.echo("No image was generated.", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@click.command("svg")
@click.argument("prompt", nargs=-1, required=True)
@click.option("-o", "--output", help="Output file path (default: auto-generated)")
@click.option(
    "-m", "--model",
    type=click.Choice(["flash", "pro"]),
    default="pro",
    help="Model: 'flash' (fast) or 'pro' (better quality)"
)
@click.option("--debug", is_flag=True, help="Debug logging")
def svg_cmd(prompt, output, model, debug):
    """Generate SVG graphics using Gemini.

    Examples:
        llmx svg "momentum arrow icon for physics game"
        llmx svg "simple cart with wheels" -o cart.svg
        llmx svg "Newton cradle diagram" -m pro
    """
    configure_logger(debug=debug)

    prompt_text = " ".join(prompt)

    try:
        from .image import generate_svg

        result = generate_svg(
            prompt=prompt_text,
            output_path=output,
            model=model,
        )

        if result:
            click.echo(f"SVG saved: {result}")
        else:
            click.echo("No SVG was generated.", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Vision Analysis Command
# ============================================================================

@click.command("vision")
@click.argument("files", nargs=-1, required=True)
@click.option("-p", "--prompt", default="Describe what you see in detail.", help="Analysis prompt")
@click.option(
    "-m", "--model",
    type=click.Choice(["flash", "pro"]),
    default="flash",
    help="Model: 'flash' (fast, default) or 'pro' (better quality)"
)
@click.option("--sample", type=int, help="Sample N frames evenly (for many images)")
@click.option("--json", "json_output", is_flag=True, help="Request JSON output")
@click.option("-o", "--output", type=click.Path(), help="Write result to FILE instead of stdout")
@click.option("--debug", is_flag=True, help="Debug logging")
def vision_cmd(files, prompt, model, sample, json_output, output, debug):
    """Analyze images or videos with Gemini vision.

    Examples:
        llmx vision screenshot.png -p "What UI issues do you see?"
        llmx vision frame*.png -p "Summarize gameplay" --sample 5
        llmx vision gameplay.mp4 -p "List all UI elements visible"
        llmx vision img1.png img2.png -p "Compare these two images"
        llmx vision report.png -p "Extract table" -o result.md
    """
    configure_logger(debug=debug)

    # Expand glob patterns and collect files
    from pathlib import Path
    import glob as glob_module

    file_list = []
    for pattern in files:
        # Check if it's a glob pattern
        if '*' in pattern or '?' in pattern:
            matches = sorted(glob_module.glob(pattern))
            file_list.extend(matches)
        else:
            file_list.append(pattern)

    if not file_list:
        click.echo("Error: No files found matching the patterns.", err=True)
        sys.exit(1)

    logger.info(f"Analyzing {len(file_list)} file(s)")

    try:
        from .vision import analyze_media, analyze_frames

        # Use analyze_frames if multiple images and sample requested
        if sample and len(file_list) > 1:
            result = analyze_frames(file_list, prompt, model, sample)
        else:
            result = analyze_media(file_list, prompt, model, json_output)

        if output:
            from pathlib import Path
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result or "")
            if not result:
                click.echo(f"[llmx:WARN] vision returned empty result — wrote 0 bytes to {output}", err=True)
        else:
            click.echo(result)

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# ============================================================================
# Deep Research Command
# ============================================================================

@click.command("research")
@click.argument("prompt", nargs=-1, required=True)
@click.option(
    "--mini",
    is_flag=True,
    help="Use o4-mini-deep-research (faster, cheaper)",
)
@click.option(
    "--max-tool-calls",
    type=int,
    help="Limit total tool calls (controls cost/latency)",
)
@click.option(
    "--code-interpreter",
    is_flag=True,
    help="Enable code interpreter for data analysis",
)
@click.option("-o", "--output", help="Save report to file (markdown)")
@click.option("--debug", is_flag=True, help="Debug logging")
def research_cmd(prompt, mini, max_tool_calls, code_interpreter, output, debug):
    """Deep research using OpenAI o3/o4-mini.

    Searches hundreds of sources and produces a detailed report with citations.
    Runs in background mode (typically takes 2-10 minutes).

    Examples:
        llmx research "economic impact of semaglutide on healthcare"
        llmx research --mini "compare React vs Svelte in 2026"
        llmx research "analyze CRISPR patent landscape" -o report.md
        llmx research --code-interpreter "global EV adoption trends with data"
    """
    configure_logger(debug=debug)

    prompt_text = " ".join(prompt)
    model = "o4-mini" if mini else "o3"

    try:
        from .research import research
        research(
            prompt=prompt_text,
            model=model,
            max_tool_calls=max_tool_calls,
            code_interpreter=code_interpreter,
            output_file=output,
            debug=debug,
        )
    except KeyboardInterrupt:
        logger.info("Research cancelled by user")
        sys.exit(130)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Text Chat Command (default behavior)
# ============================================================================

@click.command("chat")
@click.argument("prompt", nargs=-1, required=False)
@click.option(
    "-m",
    "--model",
    help="Model: 'gpt-5.5', 'claude-sonnet-4-6', 'gemini-3.1-pro-preview', 'kimi-k2.5-thinking', 'cerebras/qwen-3-coder-480b'.",
)
@click.option(
    "-p",
    "--provider",
    default=None,
    help="Provider: openai, anthropic, kimi, cerebras, google, gemini-cli, codex-cli",
)
@click.option(
    "-t",
    "--temperature",
    default=None,
    type=float,
    help="Temperature 0-2 (default 0.7).",
)
@click.option(
    "-e", "--reasoning-effort", "--effort",
    type=click.Choice(["none", "minimal", "low", "medium", "high", "xhigh"], case_sensitive=False),
    help="Thinking effort override. GPT-5.5 supports up to xhigh; llmx defaults GPT-5.5 to high on API fallback.",
)
@click.option(
    "--stream/--no-stream",
    default=False,
    help="Stream output (default: off)",
)
@click.option(
    "--compare",
    is_flag=True,
    help="Compare multiple providers",
)
@click.option(
    "--providers",
    help="Providers for --compare (default: google,openai,xai)",
)
@click.option(
    "--timeout",
    default=300,
    type=int,
    help="Timeout seconds (default: 300)",
)
@click.option("--debug", is_flag=True, help="Debug logging")
@click.option("--json", "json_output", is_flag=True, help="JSON output")
@click.option(
    "--list-providers",
    "list_providers_flag",
    is_flag=True,
    help="List providers",
)
@click.option(
    "--no-thinking",
    is_flag=True,
    help="Disable thinking/reasoning: Kimi switches to instruct model",
)
@click.option(
    "--use-old",
    is_flag=True,
    help="Use previous model version (e.g., Kimi K2 instead of K2.5)",
)
@click.option(
    "--fast",
    is_flag=True,
    help="Use fast model variant: Gemini Flash with low reasoning effort",
)
@click.option(
    "--search",
    is_flag=True,
    help="Ground response with web search (google, anthropic, xai)",
)
@click.option(
    "-s", "--system",
    help="System message for the model",
)
@click.option(
    "-f", "--file",
    "file_path",
    type=click.Path(exists=True),
    multiple=True,
    help="Read file contents as context (repeatable: -f a.md -f b.md). Prepended to prompt.",
)
@click.option(
    "--schema",
    "schema_path",
    type=click.Path(exists=True),
    help="JSON schema file for structured output",
)
@click.option(
    "--max-tokens",
    "max_tokens",
    type=int,
    default=None,
    help="Max output tokens (Gemini defaults to 8K without this — set 65536 for long outputs).",
)
@click.option(
    "-o", "--output",
    "output_path",
    type=click.Path(),
    help="Write output to file (unbuffered — no shell redirect needed).",
)
@click.option(
    "--fallback",
    "fallback_model",
    help="Fallback model on rate-limit/timeout (e.g., gemini-3-flash-preview). Auto-retries once.",
)
@click.option(
    "--lite",
    type=click.Choice(["bare", "research"], case_sensitive=False),
    default=None,
    help=(
        "Cost-saving CLI mode. 'bare' = no MCPs/tools/skills. "
        "'research' = research MCP only (papers, preprints, verify_claim). "
        "Routes openai→codex-cli too. Empty cwd, no project context."
    ),
)
@click.pass_context
def chat_cmd(
    ctx,
    prompt,
    provider,
    model,
    temperature,
    reasoning_effort,
    stream,
    compare,
    providers,
    timeout,
    debug,
    json_output,
    list_providers_flag,
    no_thinking,
    use_old,
    fast,
    search,
    system,
    file_path,
    schema_path,
    max_tokens,
    output_path,
    fallback_model,
    lite,
):
    """Text generation with LLMs (default command)."""
    configure_logger(debug=debug, json_mode=json_output)

    # List providers
    if list_providers_flag:
        provider_list = list_providers()
        if json_output:
            click.echo(json.dumps({"providers": provider_list}, indent=2))
        else:
            click.echo("Available providers:")
            for p in provider_list:
                click.echo(f"  - {p}")
        return

    # Get prompt from args or stdin
    prompt_text = " ".join(prompt) if prompt else None
    stdin_text = None
    file_text = None

    # Read from stdin if available — but only if we don't already have a prompt or file.
    # When launched by agents (Bash tool), stdin is a pipe but empty — read() blocks forever.
    if not sys.stdin.isatty() and not prompt_text and not file_path:
        logger.debug("Reading from stdin (no prompt provided)")
        stdin_text = sys.stdin.read().strip()
        logger.debug(f"Read {len(stdin_text)} chars from stdin")

    # Read from file(s) if specified — -f is repeatable, file_path is a tuple
    if file_path:
        file_parts = []
        for fp in file_path:
            try:
                part = Path(fp).read_text().strip()
            except UnicodeDecodeError as e:
                suffix = Path(fp).suffix.lower()
                hint = ""
                if suffix == ".pdf":
                    hint = (
                        "\n  PDF detected. For text-only: convert first with "
                        "`uvx 'markitdown[pdf]' FILE.pdf > FILE.md` then pass FILE.md via -f.\n"
                        "  For visual/layout: convert pages to PNG with "
                        "`pdftoppm -png -r 200 FILE.pdf out` then use `llmx vision out-*.png`."
                    )
                elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".mp4", ".mov"}:
                    hint = f"\n  Image/video detected. Use `llmx vision {fp}` instead — -f is for text context only."
                else:
                    hint = "\n  -f expects UTF-8 text. Convert binary files to text first (e.g. markitdown, pdftotext, base64)."
                click.echo(f"Error: cannot read {fp} as text ({e}).{hint}", err=True)
                sys.exit(1)
            logger.debug(f"Read {len(part)} chars from {fp}")
            file_parts.append(part)
        file_text = "\n\n".join(file_parts)
        logger.debug(f"Combined {len(file_path)} file(s): {len(file_text)} chars total")

    # Combine context sources: file > stdin > prompt
    context_parts = [p for p in [file_text, stdin_text] if p]
    if context_parts and prompt_text:
        prompt_text = "\n\n".join(context_parts) + f"\n\n{prompt_text}"
        logger.debug("Combined context with prompt")
    elif context_parts:
        prompt_text = "\n\n".join(context_parts)
    elif not prompt_text:
        click.echo(ctx.get_help())
        return

    if not prompt_text:
        logger.error("Empty prompt")
        click.echo("Error: Empty prompt.", err=True)
        sys.exit(1)

    # Load JSON schema if specified
    schema = None
    if schema_path:
        schema = json.loads(Path(schema_path).read_text())
        logger.debug(f"Loaded JSON schema from {schema_path}")

    user_specified_temp = temperature is not None
    final_temperature = temperature if temperature is not None else 0.7

    if timeout < 1 or timeout > 900:
        logger.error(f"Invalid timeout: {timeout}")
        click.echo("Error: Timeout must be between 1 and 900 seconds.", err=True)
        sys.exit(1)

    _output_file = None
    _original_stdout = None
    _result_text = None
    try:
        if compare:
            provider_list = (
                providers.split(",") if providers else ["google", "openai", "xai"]
            )
            provider_list = [p.strip() for p in provider_list]

            logger.info(f"Comparing {len(provider_list)} providers", {"providers": provider_list})
            compare_providers(
                prompt_text,
                provider_list,
                final_temperature,
                reasoning_effort,
                debug,
                json_output,
                use_old,
                user_specified_temp,
                timeout,
                search,
            )
            return

        final_provider = provider

        # --fast: override to Gemini Flash with low reasoning effort
        if fast:
            if not provider:
                final_provider = "google"
            if not model:
                from .providers import PROVIDER_CONFIGS
                fast_provider = final_provider or "google"
                fast_model = PROVIDER_CONFIGS.get(fast_provider, {}).get("flash_model")
                if fast_model:
                    model = fast_model
                    logger.info(f"--fast: using {model}")
                else:
                    logger.info(f"--fast: no fast model for {fast_provider}, using default")
            if not reasoning_effort:
                reasoning_effort = "low"
                logger.info(f"--fast: reasoning_effort=low")

        if model and not provider:
            inferred = infer_provider_from_model(model)
            if inferred:
                final_provider = inferred
                logger.debug(f"Inferred provider '{final_provider}' from model '{model}'")
            else:
                final_provider = "google"
                logger.debug(f"Could not infer provider from model '{model}', using default: google")
        elif not provider:
            final_provider = "google"

        # --no-thinking: switch to non-thinking model variant
        if no_thinking:
            if final_provider == "kimi" and not model:
                model = "kimi-k2-thinking"  # Fall back to K2 non-thinking instruct
                logger.info(f"Switching to non-thinking model: {model}")
            elif final_provider == "kimi" and model and "k2.5" in model.lower():
                model = "kimi-k2-thinking"  # K2 instruct as non-thinking fallback
                logger.info(f"Switching to non-thinking model: {model}")
            else:
                logger.warn(f"--no-thinking has no effect for provider {final_provider}")

        # --output: When streaming is explicitly enabled, chunks are tee'd to the
        # file via _TeeWriter.  When streaming is off (the default), the response
        # text returned from chat() is written to the file directly.  This avoids
        # forcing streaming (which breaks reasoning models whose delta.content is
        # empty during the thinking phase) while still capturing output reliably.

        requested_reasoning_effort = reasoning_effort
        cli_provider = preferred_cli_provider(final_provider, lite=lite)
        cli_fallback_reason = None

        if cli_provider:
            logical_provider = (
                CLI_PROVIDERS[cli_provider]["api_fallback"]
                if final_provider in CLI_PROVIDERS
                else final_provider
            )
            planned_model = model or get_model_name(logical_provider, None, use_old)
            cli_fallback_reason = needs_api_fallback(
                cli_provider, schema, system, search, stream, requested_reasoning_effort, max_tokens
            )
            if cli_fallback_reason:
                planned_transport = f"{CLI_PROVIDERS[cli_provider]['api_fallback']}-api"
            else:
                planned_transport = cli_provider
        else:
            planned_model = get_model_name(final_provider, model, use_old)
            planned_transport = f"{final_provider}-api"

        effective_reasoning_effort = requested_reasoning_effort
        reasoning_effort_source = "user" if requested_reasoning_effort else None

        if planned_transport.endswith("-api"):
            restriction = get_model_restriction(planned_model)
            if not effective_reasoning_effort and restriction and restriction.get("reasoning_effort"):
                default_effort = restriction.get("default_effort")
                if default_effort:
                    effective_reasoning_effort = default_effort
                    reasoning_effort_source = "api-default"
                else:
                    effective_reasoning_effort = "provider-default"
                    reasoning_effort_source = "provider-default"
        else:
            if not effective_reasoning_effort:
                effective_reasoning_effort = "cli-default"
                reasoning_effort_source = "cli-default"
            else:
                reasoning_effort_source = "user-requested-cli-may-ignore"

        log_payload = {
            "provider": final_provider,
            "transport": planned_transport,
            "model": planned_model,
            "stream": stream,
            "requested_reasoning_effort": requested_reasoning_effort,
            "effective_reasoning_effort": effective_reasoning_effort,
            "reasoning_effort_source": reasoning_effort_source,
        }
        if cli_fallback_reason:
            log_payload["cli_fallback_reason"] = cli_fallback_reason

        # --output: tee stdout to file (unbuffered)
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            _output_file = open(output_path, "w")
            _original_stdout = sys.stdout
            sys.stdout = _TeeWriter(sys.stdout, _output_file)

        if lite:
            log_payload["lite"] = lite
        logger.info("Starting chat", log_payload)
        _result_text = chat(
            prompt_text,
            final_provider,
            model,
            final_temperature,
            reasoning_effort,
            stream,
            debug,
            json_output,
            use_old,
            user_specified_temp,
            timeout,
            search,
            system=system,
            schema=schema,
            max_tokens=max_tokens,
            lite=lite,
        )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except SearchUnavailableError as error:
        click.echo(f"⚠ {error}", err=True)
        sys.exit(5)  # exit 5 = model-error (search not available for this model)
    except QuotaError as error:
        # Quota exhaustion is permanent — no fallback, no retry, just tell the user clearly
        click.echo(error.diagnostic_line(), err=True)
        click.echo(f"Error: {error}", err=True)
        sys.exit(error.exit_code)
    except (RateLimitError, TimeoutError_) as error:
        # Emit structured diagnostic
        click.echo(error.diagnostic_line(), err=True)

        # Try fallback if configured
        if fallback_model:
            click.echo(f"[llmx:FALLBACK] {error.error_type} → retrying with {fallback_model}", err=True)
            logger.info(f"Falling back to {fallback_model} after {error.error_type}")
            try:
                fb_provider = infer_provider_from_model(fallback_model) or final_provider
                _result_text = chat(
                    prompt_text,
                    fb_provider,
                    fallback_model,
                    final_temperature,
                    reasoning_effort if fb_provider in ("openai", "google") else None,
                    stream,
                    debug,
                    json_output,
                    use_old,
                    user_specified_temp,
                    timeout,
                    search,
                    system=system,
                    schema=schema,
                    max_tokens=max_tokens,
                    lite=lite,
                )
                return  # Fallback succeeded
            except Exception as fb_error:
                click.echo(f"[llmx:FALLBACK_FAILED] {fb_error}", err=True)
                # Fall through to original error exit

        if json_output:
            click.echo(json.dumps({
                "error": error.error_type, "message": str(error),
                "exit_code": error.exit_code, "model": error.model,
            }, indent=2), err=True)
        else:
            click.echo(f"Error: {error}", err=True)
        sys.exit(error.exit_code)
    except LlmxError as error:
        click.echo(error.diagnostic_line(), err=True)
        if json_output:
            click.echo(json.dumps({
                "error": error.error_type, "message": str(error),
                "exit_code": error.exit_code, "model": error.model,
            }, indent=2), err=True)
        else:
            click.echo(f"Error: {error}", err=True)
        sys.exit(error.exit_code)
    except Exception as error:
        # Untyped error — emit generic diagnostic
        click.echo(f"[llmx:ERROR] type=unknown exit={EXIT_GENERAL}", err=True)
        if json_output:
            click.echo(json.dumps({
                "error": error.__class__.__name__,
                "message": str(error),
                "exit_code": EXIT_GENERAL,
            }, indent=2), err=True)
        else:
            click.echo(f"Error: {error}", err=True)
        sys.exit(EXIT_GENERAL)
    finally:
        if _output_file:
            sys.stdout = _original_stdout
            _output_file.close()
            # Fallback: if _TeeWriter didn't capture anything (streaming path
            # where delta.content was empty, e.g. reasoning models), write the
            # returned result text directly.
            if os.path.getsize(output_path) == 0 and _result_text:
                with open(output_path, "w") as fallback_fh:
                    fallback_fh.write(_result_text)
                    if not _result_text.endswith("\n"):
                        fallback_fh.write("\n")
                logger.debug(f"Output written to {output_path} (fallback write)")
            elif os.path.getsize(output_path) == 0:
                click.echo(f"[llmx:WARN] -o {output_path} is 0 bytes — model likely errored before producing output", err=True)
            else:
                logger.debug(f"Output written to {output_path}")


# ============================================================================
# CLI Group with Custom Dispatch
# ============================================================================

class LlmxGroup(click.Group):
    """Custom group that defaults to chat command when no subcommand is given."""

    # Flags that belong to the group itself (not chat)
    GROUP_FLAGS = {'--version', '--help', '-h'}

    def parse_args(self, ctx, args):
        # If no args, default to chat
        if not args:
            args = ['chat']
        # If first arg is a known subcommand, let Click handle it normally
        elif args[0] in self.commands:
            pass
        # If first arg is a group-level flag (--version, --help), let Click handle it
        elif args[0] in self.GROUP_FLAGS:
            pass
        # Otherwise (prompt text OR chat flags like -m, -p, -t), route to chat
        else:
            args = ['chat'] + list(args)
        return super().parse_args(ctx, args)


@click.group(cls=LlmxGroup)
@click.version_option(version=__version__)
def cli():
    """llmx - Unified LLM CLI

    Examples:
        llmx "What is 2+2?"                                  # Text generation (default)
        llmx --model gpt-5.5 "Explain Python"                # Specific model
        llmx image "a cute robot" -o robot.png               # Image generation
        llmx svg "physics arrow icon" -o arrow.svg           # SVG generation
    """
    pass


# Register subcommands
cli.add_command(chat_cmd, name="chat")
cli.add_command(image_cmd, name="image")
cli.add_command(svg_cmd, name="svg")
cli.add_command(vision_cmd, name="vision")
cli.add_command(research_cmd, name="research")

# Batch (Gemini Batch API)
from .batch_cmd import batch_group
cli.add_command(batch_group, name="batch")


# ── Keys management (macOS Keychain) ────────────────────────

@click.group("keys")
def keys_group():
    """Manage API keys in macOS Keychain."""
    pass


@keys_group.command("set")
@click.argument("key_name")
@click.option("--value", "-v", help="Key value (prompted securely if omitted)")
def keys_set(key_name, value):
    """Store an API key in macOS Keychain.

    Examples:
        llmx keys set OPENAI_API_KEY
        llmx keys set GEMINI_API_KEY --value AIza...
    """
    from .providers import _keychain_available, _keychain_set

    if not _keychain_available():
        console.print("[red]Keychain is only available on macOS.[/red]")
        raise SystemExit(1)

    if not value:
        import getpass
        value = getpass.getpass(f"Enter value for {key_name}: ")
        if not value:
            console.print("[red]No value provided.[/red]")
            raise SystemExit(1)

    if _keychain_set(key_name, value):
        console.print(f"[green]Stored {key_name} in Keychain[/green]")
    else:
        console.print(f"[red]Failed to store {key_name}[/red]")
        raise SystemExit(1)


@keys_group.command("list")
def keys_list():
    """List API keys stored in macOS Keychain."""
    from .providers import _keychain_available, _keychain_list, PROVIDER_CONFIGS

    if not _keychain_available():
        console.print("[red]Keychain is only available on macOS.[/red]")
        raise SystemExit(1)

    stored = _keychain_list()

    # Also collect all known env var names
    all_key_names = set()
    for config in PROVIDER_CONFIGS.values():
        env_var = config.get("env_var")
        if env_var:
            for var in env_var.replace(" or ", ",").split(","):
                all_key_names.add(var.strip())

    if not stored:
        console.print("No llmx keys in Keychain.")
        console.print(f"\nKnown key names: {', '.join(sorted(all_key_names))}")
        return

    console.print("[bold]Keys in Keychain:[/bold]")
    for key in sorted(stored):
        # Mask the value
        console.print(f"  {key}")

    # Show which providers are covered
    missing = all_key_names - set(stored)
    if missing:
        console.print(f"\n[dim]Not in Keychain: {', '.join(sorted(missing))}[/dim]")


@keys_group.command("delete")
@click.argument("key_name")
def keys_delete(key_name):
    """Remove an API key from macOS Keychain."""
    from .providers import _keychain_available, _keychain_delete

    if not _keychain_available():
        console.print("[red]Keychain is only available on macOS.[/red]")
        raise SystemExit(1)

    if _keychain_delete(key_name):
        console.print(f"[green]Deleted {key_name} from Keychain[/green]")
    else:
        console.print(f"[yellow]{key_name} not found in Keychain[/yellow]")


@keys_group.command("get")
@click.argument("key_name")
def keys_get(key_name):
    """Show where an API key is resolved from (env, keychain, or missing)."""
    import os
    from .providers import _keychain_get

    env_val = os.environ.get(key_name)
    kc_val = _keychain_get(key_name)

    if env_val:
        console.print(f"{key_name}: [green]env[/green] ({key_name}={env_val[:8]}...)")
    elif kc_val:
        console.print(f"{key_name}: [blue]keychain[/blue] ({kc_val[:8]}...)")
    else:
        console.print(f"{key_name}: [red]not found[/red]")


cli.add_command(keys_group, name="keys")


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
