# llmx Usage Examples

## Quick Start

### CLI (Original Usage)
```bash
# Simple query
llmx "What is 2+2?"

# Specific provider
llmx --provider openai "Explain Python"

# GPT-5 with high reasoning
llmx --provider openai --model gpt-5-pro --reasoning-effort high "Complex question"

# Compare providers
llmx --compare "Which is better: tabs or spaces?"
```

### Python API (New!)
```python
from llmx import chat

# One-liner
response = chat("What is 2+2?", provider="openai")
print(response.content)  # "4"
print(response.usage)    # Token counts
print(response.latency)  # Response time
```

## Python API Examples

### 1. Simple Chat

```python
from llmx import chat

# Default provider (Google)
response = chat("What is Python?")
print(response.content)

# Specific provider
response = chat("Explain Rust", provider="openai", model="gpt-4o")
print(response.content)

# With system message
response = chat(
    "What is 2+2?",
    system="You are a math teacher. Explain step by step.",
    provider="anthropic"
)
print(response.content)
```

### 2. Stateful Client

```python
from llmx import LLM

# Create client with default settings
llm = LLM(provider="openai", model="gpt-5-pro", temperature=0.3)

# Multiple calls with same settings
r1 = llm.chat("What is Python?")
r2 = llm.chat("What is Rust?")
r3 = llm.chat("Compare them", temperature=0.7)  # Override temp for this call

print(r1.content)
print(r2.content)
print(r3.content)
```

### 3. Batch Processing

```python
from llmx import batch

# Process multiple prompts in parallel
prompts = [
    "What is 2+2?",
    "What is 3+3?",
    "What is 4+4?",
    "What is 5+5?",
]

responses = batch(prompts, provider="google", parallel=3)

for prompt, response in zip(prompts, responses):
    print(f"{prompt} -> {response.content}")
```

### 4. Inspection & Debugging

```python
from llmx import chat
from llmx.inspect import last_request, last_response, stats, history, clear

# Make some calls
chat("Hello", provider="openai")
chat("World", provider="google")

# Inspect last request
print(last_request())
# {'provider': 'google', 'model': 'gemini-2.5-pro', 'messages': [...]}

# Inspect last response
print(last_response())
# {'content': 'World!', 'usage': {...}, 'latency': 1.2}

# Get statistics
print(stats())
# {
#   'total_calls': 2,
#   'errors': 0,
#   'success_rate': 1.0,
#   'total_tokens': 42,
#   'avg_latency': 1.5,
#   'by_provider': {
#     'openai': {'calls': 1, 'tokens': 20, 'avg_latency': 1.1},
#     'google': {'calls': 1, 'tokens': 22, 'avg_latency': 1.9}
#   }
# }

# Get call history
for call in history(limit=5):
    print(f"{call['provider']}: {call['latency']}s")

# Clear history
clear()
```

### 5. Retry with Backoff

```python
from llmx import chat
from llmx.helpers import retry

# Auto-retry on failure (useful for flaky networks)
@retry(max_attempts=3, backoff=2.0)
def flaky_call(prompt):
    return chat(prompt, provider="openai")

# If it fails, retries with 2s, 4s, 8s backoff
response = flaky_call("Your prompt")
```

### 6. Response Caching

```python
from llmx import chat
from llmx.helpers import cache

# Cache expensive calls
@cache(ttl=3600)  # Cache for 1 hour
def expensive_analysis(code):
    return chat(
        f"Analyze this code and suggest improvements:\n{code}",
        provider="gpt-5-pro"
    )

# First call - hits LLM
result1 = expensive_analysis("def foo(): pass")  # Takes 2s

# Second call with same code - returns cached result
result2 = expensive_analysis("def foo(): pass")  # Takes 0.001s

# Different code - hits LLM again
result3 = expensive_analysis("def bar(): pass")  # Takes 2s
```

### 7. Input Validation

```python
from llmx import chat
from llmx.helpers import validate_prompt

user_input = "  What is Python?  "

try:
    # Validate and clean
    prompt = validate_prompt(user_input, min_length=5, max_length=1000)
    response = chat(prompt)
    print(response.content)
except ValueError as e:
    print(f"Invalid prompt: {e}")
```

### 8. Output Formatting

```python
from llmx import chat
from llmx.helpers import format_response

# Get JSON response
response = chat(
    "Generate a JSON object with name and age for a person named Alice, age 30",
    provider="openai"
)

# Parse as JSON
data = format_response(response, format="json")
print(data["name"])  # Alice
print(data["age"])   # 30

# Format as markdown (pretty print)
response = chat("Write a markdown tutorial on Python")
format_response(response, format="markdown")  # Renders with colors
```

### 9. Multi-Provider Comparison

```python
from llmx import LLM

providers = ["google", "openai", "anthropic", "xai"]
prompt = "What is the meaning of life?"

for provider in providers:
    llm = LLM(provider=provider)
    response = llm.chat(prompt)

    print(f"\n{provider.upper()}:")
    print(f"  Response: {response.content[:100]}...")
    print(f"  Tokens: {response.usage.get('total_tokens', 'N/A')}")
    print(f"  Latency: {response.latency:.2f}s")
```

### 10. Building a REPL

```python
from llmx import LLM
from llmx.inspect import stats

llm = LLM(provider="openai", model="gpt-4o")

print("Chat started. Type 'quit' to exit, 'stats' for statistics.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    if user_input.lower() == "stats":
        print(stats())
        continue

    response = llm.chat(user_input)
    print(f"AI: {response.content}\n")

print("Goodbye!")
```

### 11. Streaming (Coming Soon)

```python
from llmx import LLM

llm = LLM(provider="openai")

# Stream response chunks as they arrive
for chunk in llm.stream("Tell me a long story"):
    print(chunk, end="", flush=True)

print()  # Newline at end
```

## Advanced Patterns

### Context Management

```python
from llmx import LLM

class ConversationContext:
    def __init__(self, provider="openai"):
        self.llm = LLM(provider=provider)
        self.history = []

    def chat(self, message):
        # Build full context
        context = "\n".join(self.history[-5:])  # Last 5 messages
        prompt = f"Context:\n{context}\n\nUser: {message}"

        response = self.llm.chat(prompt)

        # Update history
        self.history.append(f"User: {message}")
        self.history.append(f"AI: {response.content}")

        return response

# Usage
ctx = ConversationContext()
ctx.chat("Hi, I'm working on a Python project")
ctx.chat("Can you help me with async/await?")
ctx.chat("Show me an example")  # Has context from previous messages
```

### Rate Limiting

```python
import time
from llmx import chat

class RateLimiter:
    def __init__(self, calls_per_minute=10):
        self.calls_per_minute = calls_per_minute
        self.calls = []

    def wait_if_needed(self):
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]

        if len(self.calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.calls[0])
            print(f"Rate limit reached. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

        self.calls.append(now)

# Usage
limiter = RateLimiter(calls_per_minute=10)

for prompt in many_prompts:
    limiter.wait_if_needed()
    response = chat(prompt)
    process(response)
```

### Error Handling

```python
from llmx import chat
from llmx.helpers import retry

@retry(max_attempts=3, backoff=1.0, exceptions=(RuntimeError,))
def safe_chat(prompt):
    try:
        response = chat(prompt, provider="openai")
        return response.content
    except Exception as e:
        print(f"Error: {e}")
        raise RuntimeError("LLM call failed")

# Usage
try:
    result = safe_chat("Your prompt")
    print(result)
except RuntimeError:
    print("Failed after 3 retries")
```

## Testing

### Mocking for Tests

```python
from unittest.mock import patch
from llmx import chat

def test_my_function():
    # Mock the chat function
    with patch('llmx.chat') as mock_chat:
        mock_chat.return_value.content = "Mocked response"

        result = my_function_that_uses_llmx()

        assert result == "Expected output"
        mock_chat.assert_called_once()
```

## Best Practices

1. **Use caching for repeated queries**
   ```python
   from llmx.helpers import cache

   @cache(ttl=3600)
   def analyze_code(code):
       return chat(f"Analyze: {code}")
   ```

2. **Add retry for production**
   ```python
   from llmx.helpers import retry

   @retry(max_attempts=3)
   def critical_llm_call(prompt):
       return chat(prompt)
   ```

3. **Monitor usage with inspection**
   ```python
   from llmx.inspect import stats

   # After running your app
   print(stats())  # Check tokens, costs, latency
   ```

4. **Validate user input**
   ```python
   from llmx.helpers import validate_prompt

   prompt = validate_prompt(user_input, max_length=5000)
   ```

5. **Use batch for multiple similar queries**
   ```python
   from llmx import batch

   # Instead of:
   # for prompt in prompts:
   #     responses.append(chat(prompt))

   # Do:
   responses = batch(prompts, parallel=5)
   ```

## See Also

- [ENHANCEMENT_PLAN.md](./ENHANCEMENT_PLAN.md) - Full API design
- [README.md](./README.md) - Installation and quick start
