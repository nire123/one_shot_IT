"""
Regenerate the entire example figure suite into examples/figures/.

    python examples/generate_all.py

One generator per use case (channel / rate-distortion average / rate-distortion
excess / JSCC). All settings are reduced (small n / few MC draws / coarse grids)
so the whole suite runs in a few minutes on commodity hardware.
"""
import importlib
import time

MODULES = [
    "gen_channel",
    "gen_rd_average",
    "gen_rd_excess",
    "gen_jscc",
]


def main():
    for name in MODULES:
        print(f"==== {name} ====")
        t0 = time.time()
        importlib.import_module(name).main()
        print(f"  ({time.time() - t0:.1f}s)")
    print("all figures written to examples/figures/")


if __name__ == "__main__":
    main()
