# Theory

This note is self-contained: it states the **pairwise-error-probability (PEP)
error-spectrum framework** the library is built on, derives the **achievable**
and **converse** bounds as integrals of an *error spectrum* against a *kernel*,
and shows why the prior optimisation of both is **one convex program** whose form
(LP / QP / bracketing LP) is chosen by the kernel. The companion
[`ARCHITECTURE.md`](../ARCHITECTURE.md) maps every symbol here onto a class or
method; [`API.md`](API.md) is the reference.

> Notation note. Throughout, `R` is the **total** rate (nats), `M = e^R` the
> number of competing codewords, and `w₀ = e^{-R} = 1/M` the corresponding
> *threshold* on the cumulative-mass axis.

---

## 1. The objects: a metric and its spectrum

All three settings (channel coding, rate–distortion, JSCC) share one skeleton. A
random codebook of `M` candidates is drawn i.i.d. from a prior, and performance
is governed by a single scalar **metric** `m` attached to each candidate:

| setting | candidate | metric `m` | event of interest |
|---|---|---|---|
| channel | codeword `x` | `W(y\|x)` (likelihood) | decoding error |
| rate–distortion | reproduction `y` | `d(x,y)` (distortion) | distortion of best-of-`M` |
| JSCC | pair `(v,x)` | `\|V\|·W(y\|x)·P_V(v)` | joint error |

For a fixed prior `Q` the metric induces a **spectrum** — the law of the
cumulative mass that beats a given level. Writing `σ(w)` for the mass of
candidates at metric-level ≥ the `w`-quantile, the library represents this as a
monotone **staircase** (the `F`/`A`-curve, `src/fbl/F_curve.py`):

- channel/JSCC — the **success spectrum** `S(Q; w)`: the probability that the
  transmitted candidate is ranked inside the top mass `w`;
- RD — the **distortion spectrum** `A(Q; w)`: the best-of-`M` distortion CDF.

The staircase is exact and piecewise-constant in the one-shot (lifted `|X|ⁿ`)
flavour and reconstructed by the method of types in the type-based flavour; the
two are cross-checked to agree at small `n` (see [`TESTING.md`](TESTING.md)).

---

## 2. The bounds as a kernel integral

The **random-coding bound** is an integral of the spectrum against a *kernel*
`κ` determined by the ensemble. With the antiderivative `Φ` defined by
`Φ′ = 1 − ∫κ`, every bound in the library is

$$ P = 1 - c^\top \Phi(A \cdot Q), $$

where `A·Q = σ` are the cumulative staircase masses (**linear in the prior `Q`**)
and `c` collects the non-negative metric gaps `νⱼ − νⱼ₊₁`. Two instances make
this concrete.

### 2.1 Achievability — RCU⁺ (channel / JSCC, list `L=1`)

The RCU⁺ kernel is `κ(w) = eᴿ·1{w ≤ w₀}`, giving the clamped parabola

$$ \Phi(t) = t - \frac{1}{2} e^{R} t^2 \quad (t \le w_0), \qquad
   \Phi(t) = \frac{1}{2} w_0 \quad (t \ge w_0), $$

and the achievable error `P_e = 1 − eᴿ·Γ` with
`Γ = Σ (νⱼ−νⱼ₊₁)(w₀aⱼ − ½aⱼ²)` over clamped cumulative knots
`aⱼ ≤ aⱼ₋₁ + massⱼ`, `aⱼ ≤ w₀`
(`AchievabilityQP.solve_rcu_plus`; the same program is solved by the
`phi_simplex` march).

### 2.2 Achievability — general kernel (rate–distortion, excess, list `L`)

For rate–distortion the best-of-`M` antiderivative is a genuine degree-`M`
polynomial,

$$ \Phi(t) = 1 - (1-t)^{M}, $$

and for list size `L` the JSCC kernel gives `Φ` of degree `L+1`. These have **no
QP form**; the library brackets them (next section).

### 2.3 Converse — the meta-converse, as a Dirac kernel

The meta-converse optimises the prior at a **single** threshold `w₀`. In the
Φ-view this is the **Dirac kernel** `κ = δ(w − w₀)`, whose antiderivative is the
**ramp**

$$ \Phi(t) = \min(t, w_0). $$

Because the ramp is piecewise-linear, the converse program is a **linear
program** — exactly `TypeBasedChannel.optimize_prior` /
`TypeBasedJSCC.compute_converse`. Running the achievability skeleton with the
ramp objective reproduces the converse value bit-for-bit
(`AchievabilityJSCC.solve_dirac_ramp`), a built-in normalisation check.

---

## 3. Why prior optimisation is one convex program

The objective `Γ(Q) = cᵀ Φ(σ)` is **concave in `Q`**: `Φ` is concave and `σ = A·Q`
is linear, so a concave function of a linear map. Maximising a concave function
over the probability simplex is a convex program, and the **kernel chooses its
class**:

| program | kernel | `Φ(t)` | shape | solver in `fbl` |
|---|---|---|---|---|
| converse | Dirac `δ(w−w₀)` | `min(t, w₀)` (ramp) | piecewise-linear → **LP** | `optimize_prior` / `compute_converse` |
| achievability, `L=1` | `eᴿ 1{w≤w₀}` | `t − ½eᴿt²` (clamped) | piecewise-quadratic → **QP** | `AchievabilityQP`, `AchievabilityJSCC` |
| achievability, general | positive kernel | `1−(1−t)ᴹ` / degree `L+1` | concave, non-polynomial → **bracket** | `AchievabilityLP_RD`, `ExcessRD` |

This is the core message: **the converse already optimises the prior point-wise**
(one threshold ⇒ LP), but the **achievability** bound is a kernel integral of the
*whole* spectrum, so its prior optimisation is global — and it is still convex,
exactly a QP (RCU⁺) or a certified bracketing LP (everything else).

### 3.1 The bracketing LP and its certified gap

For a non-polynomial concave `Φ`, replace it on a mesh of `K+1` knots by

- **secant** chords (which lie **below** `Φ`) → an over-estimate of `P` → a
  *certified upper bound* `P_hi`;
- **tangent** lines (which lie **above** `Φ`) → an under-estimate → a *lower
  bound* `P_lo`.

Both are LPs; the truth is sandwiched `P_lo ≤ inf_Q P ≤ P_hi` with a gap that
shrinks as `O(1/K²)` (mesh refinement). `K ≈ 20–48` is plenty in practice.
See `AchievabilityLP_RD.solve_bracketing_lp` and `AchievabilityQP.solve_bracketing_lp`.

---

## 4. The simplex march (`phi_simplex`)

The convex program is solved **directly on the simplex**, without cvxpy, for *any*
kernel and *any* setting. The gradient of `Γ` is the analytic **water-fill**
gradient

$$ g(x) = \frac{\partial \Gamma}{\partial Q(x)}
   = \sum_y \sum_{i \text{ fed by } x} \mathrm{ratio}_i \sum_{j \ge i} c_j \Phi'(\sigma_j)
   = A^{\!\top}\!\big(c \odot \Phi'(A Q)\big). $$

The prior obeys a **product of simplices** (`simplex_blocks`): one global simplex
for channel/RD, one per source-type block for JSCC (the conditional codeword-type
law). The solver marches with projected gradient (or Frank–Wolfe), projecting each
block. Optimality is the **water-filling KKT condition** — `g(x)` flat on the
support and dominated off it, *per block* — checked intrinsically by
`check_kkt` (no second solver needed).

`phi_simplex` is the production achievability optimizer: scalable, cvxpy-free,
warm-startable across a sweep, and exact for any kernel (no bracketing gap). Its
limitation is precision: `Φ` is flat past the clamp, so first-order convergence is
fast to engineering accuracy but stalls near machine precision; the QP /
bracketing-LP solvers remain as exact validation anchors, and an active-set finish
(future work) would close the last gap. (The earlier channel-only `DirectPriorOpt`
prototype has been removed in favour of this general implementation.)

---

## 5. The three settings, specialised

**Channel.** Free rate knob `M = e^{nR}` (per-symbol `M₁ = eᴿ`). The
prior is a law over input **types**; the achievability gain over optimised
memoryless is a **low-rate + large-`n` corner** effect of constant composition —
small at `n=6`, tens of percent by `n=20`.

**Rate–distortion.** `M` reproduction candidates; the prior is a law over
reproduction types. The kernel `Φ = 1−(1−t)ᴹ` forces the bracket. Average
distortion is barely prior-sensitive; **excess** distortion (below) is strongly
so.

**Excess distortion.** The excess probability of a size-`M` codebook is the
best-of-`M` of the **indicator** distortion `d_e = 1{d > T}`:

$$ P_\text{exc}(Q) = \sum_x P_X(x) (1 - q(x))^{M}, \qquad
   q(x) = Q_Y\{y : d(x,y) \le T\}. $$

So excess optimisation *is* the average-distortion machinery applied to `d_e` —
no new derivation (`ExcessRD` wraps `AchievabilityLP_RD`).

**JSCC.** No free rate knob: resolving the whole source pins `M = |V|ⁿ` for list
`L=1`. The matched metric `m = |V|·W(y\|x)·P_V(v)` already carries the source
structure, so for an i.i.d. source through a memoryless channel the non-product
prior gain is essentially **nil** — a declared null result.

---

## 6. The headline the framework was built to measure

Because converse and achievability optimise *different* functionals of the same
spectrum (one threshold vs. the whole integral), the **converse-optimal prior
reused for achievability** can be far from optimal. The library measures this
penalty exactly (see [`../RESULTS.md`](../RESULTS.md)): **5× (n=8) → 15× (n=20)**
worse for channel Z(0.1), **2.8×** for excess distortion, but ~1% for average
distortion and null for i.i.d. JSCC. Replacing the common heuristic "reuse the converse prior" with
the exact achievability program is the practical payoff.

---

## 7. Conventions that keep comparisons honest

- **Kernel consistency.** Compare the QP optimum against a memoryless prior
  scored with the **same** RCU⁺ kernel (`rcu_plus_from_F_curve`, threshold `1/M`),
  never against the tighter exact-RC `theory`. Mixing kernels breaks the
  `optimal ≤ memoryless` ordering.
- **The memoryless baseline** is *the optimal single-letter prior applied i.i.d.*
  Its i.i.d. extension is a feasible point of the `n`-letter program, so
  `optimal@n ≤ memoryless_optimal@n` holds **by construction** — the gain can
  never be a solver artefact.
- **No closed-form oracle** exists in general, so every quantity is
  cross-validated; [`TESTING.md`](TESTING.md) lists what each check guarantees.

### 7.1 Marginalization — the classical memoryless recipe

The textbook way to turn *any* prior into a memoryless one is to take its
**per-symbol marginal** and apply it i.i.d. For an optimized type prior `Q` this is

$$ P(x) = \mathbb{E}_{T \sim Q}\left[\frac{T(x)}{n}\right], $$

which is well-defined because the type prior is **exchangeable** — uniform within
each type class, so the marginal at every coordinate is identical. Note this is a
*different* construction from the [`memoryless_optimal`](API.md) baseline (which
*re-optimizes* a single-letter prior); marginalization simply **projects** the
`n`-letter optimum down to one symbol.

The **G2** figures (see [`../RESULTS.md`](../RESULTS.md)) include these
marginal-memoryless priors and measure their cost, finding two things. (1) The
marginal of the **achievability-optimal** prior is itself
near-optimal — a few percent — because at these blocklengths the optimum is
already close to i.i.d. (the constant-composition gain is a large-`n` corner). (2)
The **converse-optimal** prior, which is far too weak when *reused directly* for
achievability (it over-fits the single threshold), becomes a good achievability
prior **once marginalized**: marginalization discards exactly the non-product,
single-threshold structure that the achievability integral penalizes. Code path:
`marginal_input(Q, n, k)` → `memoryless_to_type_prior(·, n)` → `compute_curve`.
