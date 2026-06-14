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
| **prior-opt — achievable** (QP / bracketing LP) | ✓ | ✓ | ✓ |

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
from fbl import TypeBasedChannel
from fbl.prioropt import AchievabilityQP

W = np.array([[0.9, 0.1], [0.0, 1.0]])   # Z-channel
n = 12

# exact achievability-optimal prior (QP) at total rate R
aqp = AchievabilityQP(W, n)
res = aqp.solve_rcu_plus(R=n * 0.15)      # total rate = n * per-symbol rate
print("optimal P_e:", res["P_e_exact"])
```

See [`examples/`](examples/) for the figure suite and [`RESULTS.md`](RESULTS.md)
for the headline numbers and plots.

## Layout

```
src/fbl/            core engines (one-shot + type-based, all three settings)
src/fbl/prioropt/   prior optimization (converse LP + achievability QP/LP)
tests/              cross-check test suite
examples/           figure generators (reduced settings, fast)
ARCHITECTURE.md     design / library specification
```

## Validation

Every bound is cross-checked (there is no closed-form oracle in general):
one-shot ↔ type-based agreement at small `n`; analytic RCU expectation ↔
Monte-Carlo mean of random codes; converse ≤ achievable at every rate; and the
prior-opt programs against single-letter brute-force optima. See
[`tests/`](tests/) and [`ARCHITECTURE.md`](ARCHITECTURE.md).

## License

MIT — see [`LICENSE`](LICENSE).
