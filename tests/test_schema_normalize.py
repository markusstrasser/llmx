#!/usr/bin/env python3
"""Offline ($0) test for _normalize_schema_for_provider — the per-provider dialect fix.

One canonical JSON Schema must yield: OpenAI-strict (additionalProperties:false + all-required,
recursively) and Google (additionalProperties stripped). Run: uv run python3 tests/test_schema_normalize.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llmx.providers import _normalize_schema_for_provider  # noqa: E402

# A canonical schema exercising nested objects under properties, array `items`, and an anyOf union.
CANON = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["DETECTED", "NOT_FOUND"]},
        "spans": {
            "type": "array",
            "items": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
        "meta": {
            "anyOf": [
                {"type": "object", "properties": {"note": {"type": "string"}}},
                {"type": "null"},
            ]
        },
    },
    "required": ["verdict"],
}


def _walk_objects(node):
    """Yield every dict node in a schema tree."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk_objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_objects(v)


def test_openai_strict():
    o = _normalize_schema_for_provider(CANON, "openai")
    # top-level object: closed + every property required (order preserved)
    assert o["additionalProperties"] is False
    assert o["required"] == ["verdict", "spans", "meta"]
    # nested object under array items
    item = o["properties"]["spans"]["items"]
    assert item["additionalProperties"] is False and item["required"] == ["text"]
    # nested object inside anyOf; the {"type":"null"} branch is untouched
    branches = o["properties"]["meta"]["anyOf"]
    assert branches[0]["additionalProperties"] is False and branches[0]["required"] == ["note"]
    assert branches[1] == {"type": "null"}  # non-object: no additionalProperties injected
    # enums/types preserved verbatim
    assert o["properties"]["verdict"]["enum"] == ["DETECTED", "NOT_FOUND"]
    print("  ✓ openai   every object closed + all-required (recursive); non-objects untouched")


def test_google_strips():
    g = _normalize_schema_for_provider(CANON, "google")
    # additionalProperties must appear NOWHERE
    assert all("additionalProperties" not in n for n in _walk_objects(g))
    # required left exactly as authored (NOT expanded — that's an OpenAI-strict concern)
    assert g["required"] == ["verdict"]
    assert g["properties"]["verdict"]["enum"] == ["DETECTED", "NOT_FOUND"]
    # a schema that DID carry additionalProperties gets it dropped
    with_ap = {"type": "object", "additionalProperties": False, "properties": {}}
    assert "additionalProperties" not in _normalize_schema_for_provider(with_ap, "google")
    print("  ✓ google   additionalProperties stripped everywhere; required untouched")


def test_idempotent_and_passthrough():
    o1 = _normalize_schema_for_provider(CANON, "openai")
    o2 = _normalize_schema_for_provider(o1, "openai")
    assert o1 == o2, "openai normalization must be idempotent"
    # input is never mutated
    assert "additionalProperties" not in CANON
    # unknown provider + non-dict pass through unchanged
    assert _normalize_schema_for_provider(CANON, "anthropic") == CANON
    assert _normalize_schema_for_provider("scalar", "openai") == "scalar"
    assert _normalize_schema_for_provider(None, "google") is None
    print("  ✓ misc     idempotent · input not mutated · unknown-provider/scalar passthrough")


if __name__ == "__main__":
    print("[llmx schema-normalize test]")
    test_openai_strict()
    test_google_strips()
    test_idempotent_and_passthrough()
    print("\n  all schema-normalize tests passed ($0, offline).")
