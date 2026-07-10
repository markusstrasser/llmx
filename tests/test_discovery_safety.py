"""Keep unit-test discovery free of import-time live provider calls."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_root_contains_no_test_modules() -> None:
    root_test_modules = sorted(path.name for path in REPO_ROOT.glob("test*.py"))
    assert root_test_modules == [], (
        "root-level test modules are imported by broad unittest/pytest discovery; "
        f"move hermetic tests under tests/: {root_test_modules}"
    )
