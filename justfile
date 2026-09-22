# llmx task runner

# Reinstall globally (editable — source changes take effect immediately)
install:
    uv tool install --force --editable .

# Quick test: verify GPT and Gemini work
test:
    llmx chat -m gpt-6-astra "say OK" --timeout 15
    llmx chat -m gemini-3.8-flash --stream "say OK" --timeout 15
