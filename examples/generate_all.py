"""
Regenerate the entire example figure suite into examples/figures/.

    python examples/generate_all.py

All settings are reduced (small n / few MC draws / coarse grids) so the whole
suite runs in a couple of minutes on commodity hardware.
"""
import importlib
import time

MODULES = [
    "ex1_mc_spread",
    "ex2_prior_gap",
    "ex3_bounds_vs_exact",
    "ex4_fcurve_compare",
    "ex5_jscc_gain",
]


def main():
    for name in MODULES:
        print(f"[{name}]")
        t0 = time.time()
        mod = importlib.import_module(name)
        mod.main()
        print(f"  ({time.time() - t0:.1f}s)")
    print("all figures written to examples/figures/")


if __name__ == "__main__":
    main()
