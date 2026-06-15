#!/usr/bin/env python3
"""Refresh generated code snippets in Quarto Markdown files."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNIPPET_BLOCK_RE = re.compile(
    r"<!-- snippet: (?P<source>[^#\s]+)#(?P<tag>[^\s]+) -->\n"
    r"```(?P<lang>[^\n]*)\n"
    r".*?\n"
    r"```\n"
    r"<!-- /snippet -->",
    re.DOTALL,
)


def extract_tag(source: Path, tag: str) -> str:
    lines = source.read_text().splitlines()
    start = f"# tag::{tag}[]"
    end = f"# end::{tag}[]"
    depth = 0
    body: list[str] = []
    active = False

    for line in lines:
        stripped = line.strip()
        if stripped == start:
            depth = 1
            active = True
            continue
        if active and stripped.startswith("# tag::"):
            depth += 1
        if active and stripped.startswith("# end::"):
            if stripped == end:
                depth -= 1
                if depth == 0:
                    return "\n".join(body).rstrip()
            else:
                depth -= 1
            if depth < 1:
                continue
        if active:
            body.append(line)

    raise ValueError(f"Missing tag {tag!r} in {source}")


def refresh_file(path: Path) -> bool:
    original = path.read_text()

    def replace(match: re.Match[str]) -> str:
        source = match.group("source")
        tag = match.group("tag")
        lang = match.group("lang")
        snippet = extract_tag(ROOT / source, tag)
        return (
            f"<!-- snippet: {source}#{tag} -->\n"
            f"```{lang}\n"
            f"{snippet}\n"
            "```\n"
            "<!-- /snippet -->"
        )

    updated = SNIPPET_BLOCK_RE.sub(replace, original)
    if updated != original:
        path.write_text(updated)
        return True
    return False


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

    changed = 0
    for path in sorted(ROOT.rglob("*.qmd")):
        if refresh_file(path):
            changed += 1
    print(f"Refreshed snippets in {changed} file(s).")


if __name__ == "__main__":
    main()
