# API reference

Hand-written reference for the public surface of `fbl`. Grouped by layer
(see [`ARCHITECTURE.md`](../ARCHITECTURE.md) for how they fit together and
[`THEORY.md`](THEORY.md) for what the bounds mean). Signatures are the actual ones
in `src/`; method docstrings carry the finer print.

```python
import numpy as np
from fbl import (OneShotChannel, OneShotRD, OneShotJSCC,
                 TypeBasedChannel, TypeBasedRD, TypeBasedJSCC)
from fbl.prioropt import phi_view, phi_simplex          # the Φ-view mechanism
from fbl.prioropt import (AchievabilityQP, AchievabilityLP_RD, AchievabilityJSCC,
                          ExcessRD, rcu_plus_from_F_curve)   # exact anchors
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
| `converse_rate_at_eps(eps)` | — | ✓ (channel) | per-symbol nats: largest rate with converse error `≤ eps`, as a **single LP** |

> ⚠️ **Return ordering.** `optimize_prior` returns `(prior, metric)`. The JSCC
> analogue `compute_converse` returns `(error_lb, Q)` — the opposite order.

`TypeBasedChannel.converse_rate_at_eps(eps)` **inverts** the converse: instead of
fixing `M` and reading the error, it fixes the error and returns the rate, in one
LP (`min w` s.t. success `≥ 1−eps`; `R = −log(w*)/n`). The achievable companion is
`AchievabilityQP.achievable_rate_at_eps` (§2.2).

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

The **Φ-view** is the mechanism: the achievability bound is `J = cᵀΦ(A·Q)` over a
product of simplices, optimised by a first-order **simplex march** (KKT-certified),
for every setting and kernel. The exact QP / bracketing-LP solvers remain as
**validation anchors** and supply the staircase `_blocks` the march reads. The
converse prior is the meta-converse LP on the engines (`optimize_prior` /
`compute_converse`).

### 2.1 The Φ-view mechanism

**`fbl.prioropt.phi_view`** — the relaxation `J = cᵀΦ(A·Q)`.
- potentials `Φ` (+ derivatives `Φ′` and kernels `κ`):
  `phi_channel_exact`, `phi_rcu_plus`, `phi_rd_exact`, `phi_rd_smooth`,
  `phi_jscc_rcu`; registries `CHANNEL_KERNELS / RD_KERNELS / *_DERIV`.
- one-shot preprocess: `preprocess_channel(W, Q)`, `preprocess_rd(P_X, d, Q)` →
  `dict(A, c, blocks)`; `J_formula(pre,M,kernel,setting)` (= `cᵀΦ(A·Q)`) and
  `J_direct(...)` (independent quadrature).
- type-based evaluators (read the engine staircase): `J_typebased_channel(W,n,Q,M,kernel)`,
  `J_typebased_rd(P_X,d,n,Q,M,kernel)`, `J_typebased_jscc(P_V,W,n,Q_cond,M)`.

**`fbl.prioropt.phi_simplex`** — the achievability optimizer (the march).
- `build_program(setting, *, W=, P_X=, d=, P_V=, n=, M=, kernel=)` for
  `setting ∈ {"channel","rd","jscc"}` → prog dict
  `(blocks, num_q, phi, dphi, sense, simplex_blocks)`.
- `optimize(prog, M, method="pgd"|"fw", max_iter=5000, tol=1e-11, warm_start=None)`
  → `dict(Q, J, gap, iters, sense, kkt)`. Objective `J` is success for
  channel/JSCC (`P_e = 1 − J`, maximised) and distortion for RD (minimised).
- `check_kkt(prog, Q, M)` → water-filling certificate
  `dict(kkt, stationary, dual_feasible, support_spread, off_support_excess, fw_gap)`,
  evaluated **per simplex block** (one global block for channel/RD; one per
  source-type block for JSCC).
- wrappers: `optimize_channel(W,n,M,...)`, `optimize_rd(P_X,d,n,M,...)`,
  `optimize_jscc(P_V,W,n,M,...)`.

### 2.2 Exact solvers — validation anchors

**`AchievabilityQP(W_single, n)`** — channel.
- `solve_rcu_plus(R)` → `dict(Q_opt, P_e_exact, Gamma, status)` (exact RCU⁺ QP).
- `achievable_rate_at_eps(eps)` → per-symbol nats: largest rate with RCU⁺ error
  `≤ eps`, as a **single convex program** (`min w` s.t. success `≥ 1−eps`; the
  `w`-dependence is quad-over-linear, so it's jointly convex). The achievable
  companion to `converse_rate_at_eps`; both invert the bound for the `R`-vs-`n` view.
- `solve_bracketing_lp(R, kernel, K, side)` → bracket dict. `_blocks()` → staircase.

**`AchievabilityLP_RD(P_X_single, d_single, n)`** — rate-distortion.
- `solve_bracketing_lp(M, K=64)` → `dict(D_lo, D_hi, gap, Q_lo, Q_hi)`.
- `exact_D_rand(Q, M)` → true best-of-`M` distortion at a type prior. `_blocks()`.

**`AchievabilityJSCC(P_V, W, n)`** — JSCC.
- `solve_rcu_plus(M)` → `(P_e_plus, Q_opt)` exact QP (`L=1`).
- `solve_dirac_ramp(M)` → `(P_e, Q)`; **equals** `TypeBasedJSCC.compute_converse`.
- `memoryless_optimal(M)` → `(P_e@n, Q1)`; `bound_at_Q(M, Q_cond)`; `_blocks`.

**`ExcessRD(P_X_single, d_single, d_th, n)`** — excess as `d_e = 1{d > d_th}`.
- `solve_bracketing_lp(M, K=96)`, `exact_P_exc(Q_type, M)`.

**`rcu_plus_from_F_curve(knots, F_repo, w_max)`** and **`marginal_input(Q_T, n, k)`**
live in `fbl.type_based_utils` (re-exported from `fbl.prioropt`): the RCU⁺ bound
from a precomputed F-curve, and the per-symbol marginal of a type prior.

---

## 3. Minimal examples

```python
import numpy as np
from fbl.prioropt import phi_simplex as ps

# Channel: achievability-optimal prior via the simplex march (exact kernel)
W = np.array([[0.9, 0.1], [0.0, 1.0]])            # Z-channel
M = float(np.exp(12 * 0.30))                       # codebook size at n=12
prog = ps.build_program("channel", W=W, n=12, kernel="exact")
res = ps.optimize(prog, M)                         # dict(Q, J, kkt, ...)
P_e, Q_opt, certified = 1 - res["J"], res["Q"], res["kkt"]["kkt"]

# Rate-distortion (exact best-of-M kernel) and JSCC (conditional prior)
ps.optimize_rd(P_X=[0.75, 0.25], d=[[0, 1], [1, 0]], n=8, M=8.0)
ps.optimize_jscc(P_V=[0.9, 0.1], W=W, n=6, M=2.0 ** 6)
```

For the full figure suite see [`../examples/`](../examples/); for the headline
numbers see [`../RESULTS.md`](../RESULTS.md).
