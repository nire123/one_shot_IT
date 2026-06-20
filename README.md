# fbl — Finite-Blocklength Bounds with Prior Optimization

A small, **validated** Python library for one-shot / finite-blocklength
information theory, built on the **pairwise-error-probability (PEP) error-spectrum
framework**. It covers three settings and two implementation flavours, and — its
distinguishing feature — computes the **prior-optimised** achievability bound as
an *exact convex program*, not just a heuristic.

| | Channel coding | Rate–distortion | JSCC |
|---|---|---|---|
| **one-shot** (exact, `\|X\|^n`, +Monte-Carlo) | ✓ | ✓ | ✓ |
| **type-based** (poly-`n`, method of types) | ✓ | ✓ | ✓ |
| **achievable** bound | ✓ | ✓ | ✓ |
| **converse** (meta-converse LP) | ✓ | ✓ | ✓ |
| **prior-opt — converse** (LP) | ✓ | ✓ | ✓ |
| **prior-opt — achievable** (Φ-view simplex march, KKT-certified) | ✓ | ✓ | ✓ |

## Why this library

The converse meta-converse already optimises the prior point-wise (an LP at one
threshold). The **achievability** random-coding bound is a kernel integral of the
whole error spectrum, so its prior optimisation is a *global* problem. This
library implements the result that it is nonetheless an **exact convex program**:

- **channel / JSCC, `RCU⁺` kernel** → an exact convex **quadratic program**;
- **any positive kernel (incl. rate-distortion)** → a **bracketing linear program**
  with a certified, mesh-controlled gap.

This lets you replace the common heuristic of reusing the *converse*-optimal prior
for achievability, and measure exactly how much the prior family costs you.

## Install

```bash
pip install -e .            # core (numpy, scipy, cvxpy)
pip install -e ".[plots,test]"   # + matplotlib, pytest
```

## Quickstart

```python
import numpy as np
from fbl.prioropt import phi_simplex as ps

W = np.array([[0.9, 0.1], [0.0, 1.0]])     # Z-channel
n = 12
M = float(np.exp(n * 0.25))                 # codebook size at per-symbol rate 0.25

# achievability-optimal prior via the Φ-view simplex march (exact kernel)
prog = ps.build_program("channel", W=W, n=n, kernel="exact")
res = ps.optimize(prog, M)                  # first-order march on the simplex
print("optimal P_e:", 1 - res["J"])         # success J -> error
print("KKT-certified optimal:", res["kkt"]["kkt"])
print("optimal type prior:", res["Q"])
```

See [`examples/`](examples/) for the figure suite and [`RESULTS.md`](RESULTS.md)
for the headline numbers and plots.

## Documentation

The repo is meant to be read top to bottom on its GitHub page:

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — design + the **architecture diagram** and
  the unified `Φ`-view of the algorithms ([`docs/architecture.png`](docs/architecture.png),
  [`docs/algorithms.png`](docs/algorithms.png)).
- [`docs/THEORY.md`](docs/THEORY.md) — the PEP error-spectrum framework, derived
  self-contained; why each kernel gives an LP / QP / bracketing LP.
- [`docs/API.md`](docs/API.md) — every public class and method.
- [`docs/TESTING.md`](docs/TESTING.md) — what each cross-check guarantees.
- [`RESULTS.md`](RESULTS.md) + [`results/`](results/) — figures and headline numbers.

## Layout

```
src/fbl/            core engines (one-shot + type-based, all three settings)
src/fbl/prioropt/   prior optimization (Φ-view march + KKT; exact QP/LP anchors)
tests/              cross-check test suite
examples/           figure generators (reduced settings, fast)
docs/               THEORY · API · TESTING + architecture diagrams (*.mmd → *.png)
ARCHITECTURE.md     design / library specification
```

## Validation

Every bound is cross-checked (there is no closed-form oracle in general):
one-shot ↔ type-based agreement at small `n`; analytic RCU expectation ↔
Monte-Carlo mean of random codes; converse ≤ achievable at every rate; and the
prior-opt programs against single-letter brute-force optima. See
[`docs/TESTING.md`](docs/TESTING.md) for what each check guarantees
(**107 tests**), and [`tests/`](tests/) for the code.

## License

MIT — see [`LICENSE`](LICENSE).
