#!/usr/bin/env python3
"""Cross-model review of drop-litellm plan using native SDKs directly.
Proves the SDKs work for the exact use case (model-review dispatch)."""

import sys
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Collect context
PLAN = Path("/Users/alien/Projects/llmx/.claude/plans/drop-litellm.md").read_text()
PROVIDERS_PY = Path("/Users/alien/Projects/llmx/llmx/providers.py").read_text()
CLI_BACKENDS = Path("/Users/alien/Projects/llmx/llmx/cli_backends.py").read_text()
CLI_PY_EXCERPT = Path("/Users/alien/Projects/llmx/llmx/cli.py").read_text()[:8000]  # first 8K
TEST_RESULTS = """test_native_sdks.py results (15/17 passed):
- google/basic: 0.8s OK
- google/system: OK (meow)
- google/max_tokens: OK
- google/stream: 1.1s, 2 chunks OK
- google/thinking: 2.3s OK
- google/schema: OK (JSON parsed)
- google/search: OK (grounded)
- google/timeout: FAIL — min 10s deadline (trivial fix)
- openai/basic: 1.7s OK
- openai/stream: 1.5s, 21 chunks OK
- openai/system: OK
- openai/schema: OK (JSON parsed)
- xai/basic: 2.2s OK
- openrouter/basic: 0.5s OK
- kimi: FAIL — auth error (API key issue, not SDK)
"""

CONTEXT = f"""# Review Context: llmx Drop LiteLLM Plan

## Plan
{PLAN}

## Current providers.py (being replaced)
{PROVIDERS_PY}

## Current cli_backends.py (being extended)
{CLI_BACKENDS}

## CLI entry point (excerpt)
{CLI_PY_EXCERPT}

## Test Results
{TEST_RESULTS}
"""

SYSTEM = """You are reviewing a migration plan. Be concrete. No platitudes. Reference specific code, configs, and findings. Today is 2026-03-07.

Focus on:
1. What will break that the plan doesn't account for
2. Missing edge cases in the SDK migration
3. DX improvements that are missing or wrong
4. Whether the phase ordering is correct
5. Anything the author is overconfident about"""


def review_gemini(context: str) -> str:
    """Dispatch to Gemini 3.1 Pro via google-genai SDK."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        http_options=types.HttpOptions(timeout=300_000)  # 5 min
    )
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=context + "\n\nRESPOND WITH:\n## 1. What Will Break\n## 2. Missing Edge Cases\n## 3. DX Gaps\n## 4. Phase Ordering Issues\n## 5. Overconfidence Risks\n## 6. My Blind Spots",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            max_output_tokens=8192,
        ),
    )
    return response.text


def review_gpt(context: str) -> str:
    """Dispatch to GPT-5.4 via openai SDK."""
    from openai import OpenAI

    client = OpenAI(timeout=600.0)
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": context + "\n\nRESPOND WITH:\n## 1. Logical Inconsistencies\n## 2. Cost-Benefit Analysis\n## 3. Testable Predictions\n## 4. What's Missing\n## 5. My Top 5 Recommendations\n## 6. Where I'm Likely Wrong"},
        ],
        max_tokens=8192,
        reasoning_effort="high",
    )
    return response.choices[0].message.content


def main():
    print(f"Context size: {len(CONTEXT)} chars")
    print("Dispatching to Gemini 3.1 Pro + GPT-5.4 in parallel...\n")

    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(review_gemini, CONTEXT): "gemini",
            executor.submit(review_gpt, CONTEXT): "gpt",
        }
        for future in as_completed(futures):
            name = futures[future]
            start = time.time()
            try:
                text = future.result()
                print(f"[{name}] Done ({len(text)} chars)")
                results[name] = text
            except Exception as e:
                print(f"[{name}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                results[name] = f"FAILED: {e}"

    # Write outputs
    out_dir = Path("/Users/alien/Projects/llmx/.claude/plans")
    for name, text in results.items():
        out_path = out_dir / f"review-{name}.md"
        out_path.write_text(text)
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
