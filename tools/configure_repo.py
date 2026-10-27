"""Replace the GitHub username placeholder in repository metadata.

This is a maintainer utility, not part of the installed package API.

Usage::

    python tools/configure_repo.py --github-user your-name

"""

import argparse
from pathlib import Path
from typing import Iterable

PLACEHOLDER = "YOUR_GITHUB_USERNAME"
TEXT_SUFFIXES = {".md", ".toml", ".yml", ".yaml", ".py", ".cfg", ".ini", ".txt"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}


def iter_text_files(root: Path) -> Iterable[Path]:
    """Yield text files which may contain the placeholder."""
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name == "configure_repo.py":
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        yield path


def replace_placeholder(root: Path, github_user: str) -> list:
    """Replace the placeholder and return the files which changed."""
    changed = []
    for path in iter_text_files(root):
        with path.open("r", encoding="utf-8", newline="") as file:
            text = file.read()
        if PLACEHOLDER not in text:
            continue
        updated = text.replace(PLACEHOLDER, github_user)
        with path.open("w", encoding="utf-8", newline="") as file:
            file.write(updated)
        changed.append(path)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-user", required=True, help="GitHub username or organization")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (defaults to the parent of tools/)",
    )
    args = parser.parse_args()

    changed = replace_placeholder(args.root.resolve(), args.github_user)
    if not changed:
        print("No placeholder found; nothing to update.")
        return 0

    for path in changed:
        print(f"Updated {path.relative_to(args.root.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
