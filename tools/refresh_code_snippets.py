#!/usr/bin/env python3
"""Refresh generated contributors metadata before rendering the book.

Code snippets are included dynamically at render time by the
include-code-files Quarto filter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    contributors = ROOT / "book" / "contributors.txt"
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        shortlog = subprocess.check_output(
            ["git", "shortlog", "-s", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        names = []
        for line in shortlog.splitlines():
            name = line.split(maxsplit=1)[1] if line.split(maxsplit=1) else line
            if "dependabot" in name:
                continue
            if name in ["Claude", "Ju", "Hermes Agent"]:
                continue
            names.append(name)
        body = "\n".join(f"- {name}" for name in sorted(names))
        contributors.write_text(f"Contributors as of {head}:\n\n{body}\n")
    except (subprocess.CalledProcessError, IndexError):
        contributors.write_text("Contributors are listed in the repository history.\n")

    print("Updated contributors metadata.")


if __name__ == "__main__":
    main()
