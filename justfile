# llmx task runner

# Reinstall globally after code changes
install:
    uv tool install --force --reinstall .

# Quick test: verify GPT and Gemini work
test:
    llmx chat -m gpt-5.4 "say OK" --timeout 15
    llmx chat -m gemini-3-flash-preview --stream "say OK" --timeout 15
