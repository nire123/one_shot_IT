# API reference

Hand-written reference for the public surface of `fbl`. Grouped by layer
(see [`ARCHITECTURE.md`](../ARCHITECTURE.md) for how they fit together and
[`THEORY.md`](THEORY.md) for what the bounds mean). Signatures are the actual ones
in `src/`; method docstrings carry the finer print.

```python
import numpy as np
from fbl import (OneShotChannel, OneShotRD, OneShotJSCC,
                 TypeBasedChannel, TypeBasedRD, TypeBasedJSCC)
from fbl.prioropt import (AchievabilityQP, AchievabilityLP_RD, AchievabilityJSCC,
                          ExcessRD, TypeBasedBlockLP, TypeBasedBlockLPRD,
                          DirectPriorOpt, rcu_plus_from_F_curve)
```

> **Rate / `M` convention.** `R` is the **total** rate in nats, `M = e^R`
> competitors, `w₀ = 1/M`. Channel takes a free `M` (or `R`); JSCC pins
> `M = |V|ⁿ` for list `L=1`; RD uses `M` reproduction candidates.

---

## 1. Engines — bounds

The six engines are the `{channel, RD, JSCC} × {one-shot, type-based}`
permutation. **Channel and RD share a uniform interface** via the base classes
`OneShotBase` / `TypeBasedBase`; **JSCC deviates** (matrix-valued encoder
`Q_{X|V}`, different method names) — those differences are flagged inline.

### 1.1 Shared interface (channel & RD)

`OneShotChannel(W)`, `OneShotRD(P_X, d)` — exact in the lifted `|X|ⁿ` space (n is
implied by the lifted operands); `TypeBasedChannel(W, n)`, `TypeBasedRD(P_X, d, n)`
— method of types.

| method | one-shot | type-based | returns |
|---|---|---|---|
| `compute_curve(prior)` | ✓ | ✓ | the `F`/`A` staircase for `prior` |
| `theory(curve, M, num_refined_points=1000)` | ✓ | ✓ | achievable bound (float) |
| `optimize_prior(M)` | ✓ | ✓ | `(prior, metric)` — converse-optimal prior + its value |
| `draw_random_code(prior, M, rng)` | ✓ | — | a sampled codebook |
| `evaluate(code)` | ✓ | — | realised error/distortion of `code` |
| `mc(prior, M, num_trials=1000, seed=None)` | ✓ | — | Monte-Carlo mean ± spread |
| `validate(prior, M, ...)` | ✓ | — | dict incl. `within_ci` (theory ↔ MC) |
| `check_kkt(M, prior, s)` | ✓ | (channel/RD: —) | KKT optimality flags `cond1/cond2` |
| `get_one_shot_object()` | — | ✓ | the lifted `OneShot*` for cross-checking |

> ⚠️ **Return ordering.** `optimize_prior` returns `(prior, metric)`. The JSCC
> analogue `compute_converse` returns `(error_lb, Q)` — the opposite order.

### 1.2 `OneShotJSCC(P_V, W)` and `TypeBasedJSCC(P_V, W, n)`

JSCC's encoder is a conditional law `Q_{X|V}` (shape `(k_v, k_x)`), not a vector
prior, so it does not inherit the base classes.

| method | one-shot | type-based | returns |
|---|---|---|---|
| `compute_f_curve(Q_XgV)` / `compute_f_curve(P_T_VX)` | ✓ | ✓ | the JSCC error spectrum |
| `achievable_bound(M, Q_XgV \| P_T_VX)` | ✓ | ✓ | RCU⁺ achievable error (float) |
| `compute_converse(M)` | ✓ | ✓ | `(error_lb, Q)` — meta-converse LP |
| `check_kkt(M, Q_XgV, s)` | ✓ | — | KKT flags |
| `memoryless_prior(Q_XgV)` | — | ✓ | type prior from a single-letter `Q_{X|V}` |
| `q_xgv_to_type_prior(Q_XgV)` | — | ✓ | exact `n=1` type-prior bijection |
| `draw_random_code` / `evaluate` / `mc(...)` | ✓ | — | Monte-Carlo path |
| `get_one_shot()` | — | ✓ | the lifted `OneShotJSCC` |

---

## 2. `fbl.prioropt` — prior optimisation

This is the library's contribution: the **achievability** prior as an exact
convex program, and the **converse** prior as an LP. All build *additively* on
the engines (they only read their type machinery).

### 2.1 Achievability — exact convex program

**`AchievabilityQP(W_single, n)`** — channel, exact QP (RCU⁺) + bracket.
- `solve_rcu_plus(R)` → `dict(Q_opt, P_e_exact, Gamma, status)`. The exact QP;
  `inf_Q P_e = 1 − eᴿ·Γ*`.
- `solve_bracketing_lp(R, kernel="exact"|"rcu", K=64, side="both")` →
  `dict(P_lo, P_hi, gap, Q_lo, Q_hi)`. General-kernel certified bracket.

**`AchievabilityLP_RD(P_X_single, d_single, n)`** — rate–distortion bracket.
- `solve_bracketing_lp(M, K=64)` → `dict(D_lo, D_hi, gap, Q_lo, Q_hi)`; `D_hi`
  (secant) is the certified upper bound.
- `exact_D_rand(Q, M)` → true best-of-`M` distortion at a type prior (F-curve).

**`AchievabilityJSCC(P_V, W, n)`** — JSCC QP / bracket + baselines.
- `solve_rcu_plus(M)` → `(P_e_plus, Q_opt)` exact QP (`L=1`).
- `solve_bracketing_lp(M, L, K=64, side="lower")` → `(P_e, Q)` for general list `L`.
- `solve_dirac_ramp(M)` → `(P_e, Q)`; **must equal** `TypeBasedJSCC.compute_converse`
  (built-in normalisation check).
- `memoryless_optimal(M)` → `(P_e@n, Q1)` — the `n=1` optimum applied i.i.d.
  (fast, exact, the standard baseline).
- `bound_at_Q(M, Q_cond)` / `memoryless_baseline(M, ...)` — helpers.

**`ExcessRD(P_X_single, d_single, d_th, n)`** — excess distortion as the
indicator-distortion `d_e = 1{d > d_th}`; thin wrapper over `AchievabilityLP_RD`.
- `solve_bracketing_lp(M, K=96)` → bracket dict.
- `exact_P_exc(Q_type, M)` → exact excess probability at a prior.

### 2.2 The unified direct program

**`DirectPriorOpt(W_single, n)`** — first-order march on the simplex, exact for
*any* kernel (the [`THEORY.md`](THEORY.md) §4 Φ-view).
- `solve(R, kernel="rcu"|"converse", method="pgd"|"fw", warm_start=None,
  max_iter=5000, tol=1e-9)` → `dict(P_e, Q_opt, gap, iters, method, kernel)`.
- `directional_derivative(Q, mu, R, kernel="rcu")` → `⟨g, μ⟩` (needs `Σμ = 0`).

### 2.3 Converse — block LP

**`TypeBasedBlockLP(W_single, n)`** / **`TypeBasedBlockLPRD(P_X, d, n)`**
- `.solve(w_grid, ...)` → `dict(Q_opt, S_per_k, chord_integral, P_ub, lp_value,
  lp_status)`. The kink-adaptive chord-rule converse-prior LP.

**`rcu_plus_from_F_curve(knots, F_repo, w_max)`** → the RCU⁺ bound
`P(R;Q) = 1 − (1/w_max)∫₀^{w_max} S(Q;w) dw` from a precomputed F-curve. Use this
(not `theory`) to score memoryless priors against the QP — see the
kernel-consistency convention in [`THEORY.md`](THEORY.md) §7.

---

## 3. Minimal examples

```python
# Channel: exact achievability-optimal prior (QP) at total rate R = n·0.15
W = np.array([[0.9, 0.1], [0.0, 1.0]])            # Z-channel
aqp = AchievabilityQP(W, n=12)
res = aqp.solve_rcu_plus(R=12 * 0.15)
res["P_e_exact"], res["Q_opt"]                    # optimal bound + type prior

# Rate-distortion: certified bracket of the optimal reproduction prior
alp = AchievabilityLP_RD(P_X_single=[0.75, 0.25], d_single=[[0,1],[1,0]], n=6)
br = alp.solve_bracketing_lp(M=8, K=48)           # br["D_lo"] <= inf D <= br["D_hi"]

# The unified view: same optimum, no cvxpy
dp = DirectPriorOpt(W, n=12).solve(R=12 * 0.15, kernel="rcu")
```

For the full figure suite see [`../examples/`](../examples/); for the headline
numbers see [`../RESULTS.md`](../RESULTS.md).
