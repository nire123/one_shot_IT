"""Shared helpers for the example figure generators.

Run any example directly (``python examples/exN_*.py``) or all of them via
``python examples/generate_all.py``.  Figures are written to ``examples/figures/``.
All settings are intentionally *reduced* (small n / few MC draws / coarse grids)
so the whole suite renders in a couple of minutes; they are representative, not
thesis-resolution.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")            # headless
import matplotlib.pyplot as plt  # noqa: E402

# make `import fbl` work from a source checkout without installation
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")
    return path
