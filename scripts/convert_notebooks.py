"""Regenerate docs/examples/*.md from docs/examples/*.ipynb.

The notebooks are the source of truth and are shipped as-is so readers can
download and run them directly. Zensical can't render notebooks itself, so
each one is also exported to a markdown page, with a "run it yourself" note
linking back to the notebook inserted after the title.
"""

import subprocess
import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).parent.parent / "docs" / "examples"


def note_for(notebook_name: str) -> str:
    return (
        '!!! note "Run it yourself"\n'
        f"    This page is a static export of a Jupyter notebook. "
        f"[Download {notebook_name}]({notebook_name}) to run it yourself.\n"
    )


def convert(notebook: Path) -> None:
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "markdown",
            str(notebook),
            "--output-dir",
            str(EXAMPLES_DIR),
        ],
        check=True,
    )
    md_path = notebook.with_suffix(".md")
    text = md_path.read_text()
    title, _, rest = text.partition("\n\n")
    md_path.write_text(f"{title}\n\n{note_for(notebook.name)}\n{rest}")


def main() -> None:
    notebooks = sorted(EXAMPLES_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {EXAMPLES_DIR}", file=sys.stderr)
        sys.exit(1)
    for notebook in notebooks:
        convert(notebook)


if __name__ == "__main__":
    main()
