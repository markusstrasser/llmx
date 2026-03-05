#!/usr/bin/env python3
"""
llmx - Unified LLM CLI via LiteLLM

Usage:
    llmx "your prompt"
    llmx --model gpt-5.4 "your prompt"
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
    LlmxError, RateLimitError, TimeoutError_, EXIT_GENERAL,
    PROVIDER_CONFIGS, get_model_name, get_model_restriction,
)
from .cli_backends import preferred_cli_provider, needs_api_fallback, CLI_PROVIDERS
from .logger import configure_logger, logger

console = Console()

# Subcommand names for detection
SUBCOMMANDS = {"image", "svg", "vision", "research", "batch"}


# ============================================================================
# Image Generation Commands
# ============================================================================

@click.command("image")
@click.argument("prompt", nargs=-1, required=True)
@click.option("-o", "--output", help="Output file path (default: auto-generated)")
@click.option(
    "-m", "--model",
    type=click.Choice(["flash", "pro"]),
    default="pro",
    help="Model: Both use Gemini 3 Pro Image (gemini-3-pro-image-preview)"
)
@click.option(
    "--aspect-ratio", "-a",
    default="1:1",
    help="Aspect ratio: 1:1, 16:9, 9:16, 4:3, 3:4, 5:4, 4:5"
)
@click.option(
    "--resolution", "-r",
    type=click.Choice(["1K", "2K", "4K"]),
    default="1K",
    help="Resolution: 1K, 2K, 4K (2K+ requires pro model)"
)
@click.option("--debug", is_flag=True, help="Debug logging")
def image_cmd(prompt, output, model, aspect_ratio, resolution, debug):
    """Generate images using Gemini Nano Banana models.

    Examples:
        llmx image "a cute robot mascot"
        llmx image "pixel art knight with sword" -o knight.png
        llmx image "game background forest" -m pro -r 2K
        llmx image "physics momentum diagram" --aspect-ratio 16:9
    """
    configure_logger(debug=debug)

    prompt_text = " ".join(prompt)

    try:
        from .image import generate_image

        result = generate_image(
            prompt=prompt_text,
            output_path=output,
            model=model,
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
@click.option("--debug", is_flag=True, help="Debug logging")
def vision_cmd(files, prompt, model, sample, json_output, debug):
    """Analyze images or videos with Gemini vision.

    Examples:
        llmx vision screenshot.png -p "What UI issues do you see?"
        llmx vision frame*.png -p "Summarize gameplay" --sample 5
        llmx vision gameplay.mp4 -p "List all UI elements visible"
        llmx vision img1.png img2.png -p "Compare these two images"
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
    help="Model: 'gpt-5.4', 'claude-sonnet-4-6', 'gemini-3.1-pro-preview', 'kimi-k2.5-thinking', 'cerebras/qwen-3-coder-480b'.",
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
    "--reasoning-effort",
    type=click.Choice(["none", "minimal", "low", "medium", "high", "xhigh"], case_sensitive=False),
    help="Thinking effort override. GPT-5.4 supports up to xhigh; llmx defaults GPT-5.4 to high on API fallback.",
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
    default=120,
    type=int,
    help="Timeout seconds (default: 120)",
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
    help="Read file contents as context (prepended to prompt, like stdin)",
)
@click.option(
    "--schema",
    "schema_path",
    type=click.Path(exists=True),
    help="JSON schema file for structured output",
)
@click.option(
    "--fallback",
    "fallback_model",
    help="Fallback model on rate-limit/timeout (e.g., gemini-3-flash-preview). Auto-retries once.",
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
    fallback_model,
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

    # Read from stdin if available
    if not sys.stdin.isatty():
        logger.debug("Reading from stdin")
        stdin_text = sys.stdin.read().strip()
        logger.debug(f"Read {len(stdin_text)} chars from stdin")

    # Read from file if specified
    if file_path:
        file_text = Path(file_path).read_text().strip()
        logger.debug(f"Read {len(file_text)} chars from {file_path}")

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

    if timeout < 1 or timeout > 600:
        logger.error(f"Invalid timeout: {timeout}")
        click.echo("Error: Timeout must be between 1 and 600 seconds.", err=True)
        sys.exit(1)

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

        requested_reasoning_effort = reasoning_effort
        cli_provider = preferred_cli_provider(final_provider)
        cli_fallback_reason = None

        if cli_provider:
            logical_provider = (
                CLI_PROVIDERS[cli_provider]["api_fallback"]
                if final_provider in CLI_PROVIDERS
                else final_provider
            )
            planned_model = model or get_model_name(logical_provider, None, use_old)
            cli_fallback_reason = needs_api_fallback(
                cli_provider, schema, system, search, stream, requested_reasoning_effort
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

        logger.info("Starting chat", log_payload)
        chat(
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
        )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except (RateLimitError, TimeoutError_) as error:
        # Emit structured diagnostic
        click.echo(error.diagnostic_line(), err=True)

        # Try fallback if configured
        if fallback_model:
            click.echo(f"[llmx:FALLBACK] {error.error_type} → retrying with {fallback_model}", err=True)
            logger.info(f"Falling back to {fallback_model} after {error.error_type}")
            try:
                fb_provider = infer_provider_from_model(fallback_model) or final_provider
                chat(
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
        llmx --model gpt-5.4 "Explain Python"                # Specific model
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


def main():
    """Entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
