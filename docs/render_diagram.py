#!/usr/bin/env python3
"""Render the Mermaid architecture sources (``docs/*.mmd``) to PNG.

The ``.mmd`` files are the single source of truth (they are also embedded
verbatim in ``ARCHITECTURE.md`` so they render on the GitHub page). This
script produces the committed ``.png`` companions so the diagrams also show
up wherever Mermaid is not rendered.

Renderer preference
-------------------
1. Local **mermaid-cli** (``mmdc``) if on PATH — offline, best quality::

       npm install -g @mermaid-js/mermaid-cli

2. Fallback: the **mermaid.ink** web service (needs network access).

Usage
-----
    python docs/render_diagram.py                # render every docs/*.mmd
    python docs/render_diagram.py docs/architecture.mmd
"""
import base64
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

DOCS = Path(__file__).resolve().parent


def render_with_mmdc(src: Path, out: Path) -> bool:
    """Render via local mermaid-cli; return False if mmdc is not installed."""
    mmdc = shutil.which("mmdc")
    if not mmdc:
        return False
    subprocess.run(
        [mmdc, "-i", str(src), "-o", str(out), "-b", "white", "-s", "2"],
        check=True,
    )
    return True


def render_with_ink(src: Path, out: Path) -> bool:
    """Render via the mermaid.ink service (base64-of-source in the URL)."""
    code = src.read_text(encoding="utf-8")
    token = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{token}?type=png&bgColor=white"
    req = urllib.request.Request(url, headers={"User-Agent": "fbl-doc-render"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    out.write_bytes(data)
    return True


def main(argv):
    srcs = [Path(a) for a in argv] if argv else sorted(DOCS.glob("*.mmd"))
    if not srcs:
        print("no .mmd sources found", file=sys.stderr)
        return 1
    rc = 0
    for src in srcs:
        out = src.with_suffix(".png")
        try:
            if render_with_mmdc(src, out):
                print(f"[mmdc]        {src.name} -> {out.name}")
            else:
                render_with_ink(src, out)
                size = out.stat().st_size
                print(f"[mermaid.ink] {src.name} -> {out.name} ({size} bytes)")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"FAILED {src.name}: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
