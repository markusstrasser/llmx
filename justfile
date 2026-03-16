# llmx task runner

# Reinstall globally (editable — source changes take effect immediately)
install:
    uv tool install --force --editable .

# Quick test: verify GPT and Gemini work
test:
    llmx chat -m gpt-5.4 "say OK" --timeout 15
    llmx chat -m gemini-3-flash-preview --stream "say OK" --timeout 15
