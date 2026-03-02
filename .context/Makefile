# Context views for cross-model review
# Usage: make [target]  — generates .xml views from working tree
# Targets rebuild only when source files change (Make dependency tracking)

SHELL := /bin/bash
PROJECT_ROOT := $(shell cd .. && pwd)
REPOMIX := repomix --quiet --style xml --no-security-check --no-git-sort-by-changes

# Source file sets
PY_FILES := $(shell find $(PROJECT_ROOT)/llmx -name '*.py' -not -path '*/__pycache__/*' 2>/dev/null)
DOC_FILES := $(shell find $(PROJECT_ROOT) -maxdepth 1 -name '*.md' 2>/dev/null)
ALL_FILES := $(PY_FILES) $(DOC_FILES)

.PHONY: all clean filetree diffs

all: full.xml src.xml signatures.xml filetree.xml

# Full project context — source + docs + config
full.xml: $(ALL_FILES) $(PROJECT_ROOT)/pyproject.toml
	cd $(PROJECT_ROOT) && $(REPOMIX) \
		--include "llmx/**/*.py,*.md,pyproject.toml" \
		--ignore ".context/**,.claude/**,__pycache__/**" \
		-o .context/$@

# Source code only — all Python files
src.xml: $(PY_FILES)
	cd $(PROJECT_ROOT) && $(REPOMIX) \
		--include "llmx/**/*.py" \
		--ignore "__pycache__/**" \
		-o .context/$@

# Compressed signatures — function/class outlines via Tree-sitter
signatures.xml: $(PY_FILES)
	cd $(PROJECT_ROOT) && $(REPOMIX) \
		--include "llmx/**/*.py" \
		--ignore "__pycache__/**" \
		--compress \
		-o .context/$@

# Directory structure only
filetree.xml:
	cd $(PROJECT_ROOT) && $(REPOMIX) \
		--no-files --no-file-summary \
		--ignore ".context/**,.claude/**,__pycache__/**" \
		-o .context/$@

# Git diffs — current uncommitted changes
diffs.xml:
	cd $(PROJECT_ROOT) && $(REPOMIX) \
		--no-files --no-file-summary --no-directory-structure \
		--include-diffs \
		-o .context/$@

clean:
	rm -f *.xml
