#!/usr/bin/env python3
"""Test script for llmx API - verify it works for LLM-to-LLM calls"""

import sys
import os

print("=" * 60)
print("Testing llmx API for non-interactive LLM calls")
print("=" * 60)

# Test 1: Import the API
print("\n[Test 1] Importing llmx API...")
try:
    from llmx import chat, LLM, Response, batch
    from llmx.inspect import stats, last_response, last_request, clear
    from llmx.helpers import retry, cache, validate_prompt
    print("✓ Import successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Check which provider we can use
provider = None
if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
    provider = "google"
elif os.getenv("OPENAI_API_KEY"):
    provider = "openai"
elif os.getenv("ANTHROPIC_API_KEY"):
    provider = "anthropic"

if not provider:
    print("\n⚠  No API keys found. Set one of:")
    print("  - GOOGLE_API_KEY or GEMINI_API_KEY")
    print("  - OPENAI_API_KEY")
    print("  - ANTHROPIC_API_KEY")
    print("\nSkipping live API tests, but imports work! ✓")
    sys.exit(0)

print(f"\n✓ Using provider: {provider}")

# Test 2: Simple chat function
print("\n[Test 2] Testing chat() function...")
try:
    response = chat("What is 2+2? Answer with just the number.", provider=provider)
    print(f"✓ Chat successful")
    print(f"  Content: {response.content[:100]}")
    print(f"  Provider: {response.provider}")
    print(f"  Model: {response.model}")
    print(f"  Usage: {response.usage}")
    print(f"  Latency: {response.latency:.2f}s")

    # Verify Response has expected attributes
    assert hasattr(response, 'content')
    assert hasattr(response, 'provider')
    assert hasattr(response, 'model')
    assert hasattr(response, 'usage')
    assert hasattr(response, 'latency')
    assert isinstance(response.usage, dict)
    assert 'total_tokens' in response.usage
    print("✓ Response object structure valid")

except Exception as e:
    print(f"✗ Chat failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Stateful LLM client
print("\n[Test 3] Testing LLM class...")
try:
    llm = LLM(provider=provider, temperature=0.7)
    r1 = llm.chat("Say hello in one word")
    print(f"✓ LLM.chat() successful")
    print(f"  Response: {r1.content[:100]}")
    print(f"  Tokens: {r1.usage.get('total_tokens', 'N/A')}")
except Exception as e:
    print(f"✗ LLM class failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Inspection tools
print("\n[Test 4] Testing inspection tools...")
try:
    last_req = last_request()
    last_resp = last_response()
    call_stats = stats()

    print(f"✓ Inspection tools work")
    print(f"  Last request provider: {last_req.get('provider')}")
    print(f"  Last response tokens: {last_resp.get('usage', {}).get('total_tokens')}")
    print(f"  Total calls: {call_stats.get('total_calls')}")
    print(f"  Total tokens: {call_stats.get('total_tokens')}")
    print(f"  Avg latency: {call_stats.get('avg_latency', 0):.2f}s")

    # Verify stats structure
    assert call_stats['total_calls'] >= 2, "Should have at least 2 calls"
    assert 'by_provider' in call_stats
    assert provider in call_stats['by_provider']
    print("✓ Stats structure valid")
except Exception as e:
    print(f"✗ Inspection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Helper functions
print("\n[Test 5] Testing helper functions...")
try:
    # Test validation
    validated = validate_prompt("  test prompt  ", min_length=5)
    assert validated == "test prompt"
    print("✓ validate_prompt() works")

    # Test retry decorator (won't actually retry since call should succeed)
    @retry(max_attempts=2, backoff=0.5)
    def test_retry():
        return chat("Hi", provider=provider)

    result = test_retry()
    assert hasattr(result, 'content')
    print("✓ retry() decorator works")

    # Test cache decorator
    call_count = [0]

    @cache(ttl=60)
    def test_cache(prompt):
        call_count[0] += 1
        return chat(prompt, provider=provider)

    # First call - should hit LLM
    cached_result1 = test_cache("Cache test")
    assert call_count[0] == 1
    print("✓ cache() decorator works (first call)")

    # Second call with same prompt - should use cache
    cached_result2 = test_cache("Cache test")
    assert call_count[0] == 1, "Should not increment (cache hit)"
    print("✓ cache() decorator works (cache hit)")

    # Third call with different prompt - should hit LLM
    cached_result3 = test_cache("Different prompt")
    assert call_count[0] == 2, "Should increment (cache miss)"
    print("✓ cache() decorator works (cache miss)")

except Exception as e:
    print(f"✗ Helpers failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Batch processing
print("\n[Test 6] Testing batch processing...")
try:
    prompts = ["Say 'A'", "Say 'B'", "Say 'C'"]
    responses = batch(prompts, provider=provider, parallel=2)

    assert len(responses) == len(prompts)
    print(f"✓ Batch processing successful")
    print(f"  Processed {len(responses)} prompts")
    for i, resp in enumerate(responses):
        print(f"  Response {i+1}: {resp.content[:50]}")
except Exception as e:
    print(f"✗ Batch failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final stats
print("\n[Test 7] Final statistics...")
try:
    final_stats = stats()
    print(f"✓ Final statistics:")
    print(f"  Total API calls: {final_stats['total_calls']}")
    print(f"  Total tokens used: {final_stats['total_tokens']}")
    print(f"  Average latency: {final_stats['avg_latency']:.2f}s")
    print(f"  Success rate: {final_stats['success_rate']:.1%}")
    print(f"  Providers used: {list(final_stats['by_provider'].keys())}")
except Exception as e:
    print(f"✗ Final stats failed: {e}")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("llmx API is ready for LLM-to-LLM calls")
print("=" * 60)
