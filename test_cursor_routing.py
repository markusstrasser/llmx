#!/usr/bin/env python3
"""Lock the cursor-transport routing invariant.

Regression guard for the bug where `cursor/<model>` silently routed to the PAID
API (cursor/gemini-3-flash -> google, cursor/kimi-... -> kimi) because the cursor
check sat AFTER the bare-substring family checks in infer_provider_from_model.
The `cursor/` prefix is an explicit override and must win over every substring.

Run: uv run python3 test_cursor_routing.py   (or pytest)
"""
from llmx.providers import infer_provider_from_model as infer


def test_cursor_prefix_overrides_substring_families():
    # cursor/<anything> must proxy through the Cursor subscription, NOT the paid API
    # that the bare substring would otherwise select.
    for model in ("cursor/gemini-3-flash", "cursor/kimi-k2.5", "cursor/grok-4",
                  "cursor/minimax-m3", "cursor/qwen-3", "cursor/deepseek-v3",
                  "cursor/claude-opus-4-8", "cursor/gpt-5.5"):
        assert infer(model) == "cursor", f"{model} must route to cursor, got {infer(model)}"


def test_bare_composer_is_cursor():
    assert infer("composer-2.5") == "cursor"
    assert infer("composer-2.5-fast") == "cursor"


def test_no_regression_for_bare_model_names():
    # The fix must not change routing for any non-cursor model.
    expected = {
        "gemini-3-flash": "google", "kimi-k2.5": "kimi", "grok-4": "xai",
        "minimax-m3": "minimax", "qwen-3": "cerebras", "gpt-5.5": "openai",
        "claude-opus-4-8": "anthropic", "deepseek-v3": "deepseek",
        "openrouter/x": "openrouter",
    }
    for model, prov in expected.items():
        assert infer(model) == prov, f"{model}: want {prov}, got {infer(model)}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
