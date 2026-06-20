# Architecture & Library Specification

`fbl` is organised as a small dependency DAG: infra → engines → prior-opt →
examples. Three information-theoretic **settings** (channel / RD / JSCC) appear in
parallel throughout, in two implementation **flavours** (one-shot / type-based).

> **Companion docs:** [THEORY](docs/THEORY.md) (the framework and why each kernel
> gives a QP/LP/bracket) · [API](docs/API.md) (every public class & method) ·
> [TESTING](docs/TESTING.md) (what each cross-check guarantees).

## The library at a glance

The dependency DAG is four layers — **infra → engines → prior-opt → examples** —
with the three settings (channel / RD / JSCC) running in parallel through each.
The achievable bound of every setting is integrated against a *kernel*; the
prior-opt layer is the contribution, splitting cleanly into a **converse LP** and
an **achievability** exact convex program.

![fbl architecture](docs/architecture.png)

<details><summary>Mermaid source (renders on GitHub; <code>python docs/render_diagram.py</code> regenerates the PNG)</summary>

```mermaid
flowchart TB
  classDef infra  fill:#e8ecff,stroke:#4456a6,color:#1a2240;
  classDef engine fill:#e6f6e6,stroke:#3c7a3c,color:#163016;
  classDef prior  fill:#ffe9e6,stroke:#a64a3c,color:#3a1410;
  classDef ex     fill:#fff6da,stroke:#9c7a1e,color:#3a2c08;

  subgraph EX["EXAMPLES · generators"]
    direction LR
    gch["gen_channel"]:::ex
    grd["gen_rd_average<br/>gen_rd_excess"]:::ex
    gjs["gen_jscc"]:::ex
    gch ~~~ grd ~~~ gjs
  end
  subgraph PO["PRIOR OPTIMIZATION"]
    subgraph ACH["achievability — convex"]
      direction LR
      QP["AchievabilityQP<br/>channel · RCU⁺ → QP"]:::prior
      LPRD["AchievabilityLP_RD<br/>RD · kernel → bracket LP"]:::prior
      JS["AchievabilityJSCC<br/>JSCC · QP + bracket"]:::prior
      EXC["ExcessRD<br/>indicator distortion"]:::prior
      DIR["DirectPriorOpt<br/>unified Φ-view"]:::prior
      QP ~~~ LPRD ~~~ JS ~~~ EXC ~~~ DIR
    end
    subgraph CONV["converse — LP"]
      direction LR
      TBLP["TypeBasedBlockLP"]:::prior
      TBLPRD["TypeBasedBlockLPRD"]:::prior
      TBLP ~~~ TBLPRD
    end
  end
  subgraph ENG["ENGINES · bounds"]
    direction LR
    OSC["OneShotChannel"]:::engine
    OSR["OneShotRD"]:::engine
    OSJ["OneShotJSCC"]:::engine
    TBC["TypeBasedChannel"]:::engine
    TBR["TypeBasedRD"]:::engine
    TBJ["TypeBasedJSCC"]:::engine
    OSC ~~~ OSR ~~~ OSJ
    TBC ~~~ TBR ~~~ TBJ
  end
  subgraph INF["INFRA · fbl core (private)"]
    direction LR
    TCC["type_class_core<br/>type enum · indexing"]:::infra
    FC["F_curve<br/>F/A-curve integrators"]:::infra
    UT["*_utils<br/>channels · sources · prior maps"]:::infra
    TCC ~~~ FC ~~~ UT
  end
  EX ==> PO ==> ENG ==> INF
```
</details>

The same DAG as a text tree:

```
  examples/                         figure generators (reduced settings)
      ▲
  fbl.prioropt                      PRIOR OPTIMIZATION
      AchievabilityQP   (channel)   exact QP (RCU⁺) + bracketing LP
      AchievabilityLP_RD            bracketing LP (degree-M kernel)
      AchievabilityJSCC             exact QP (L=1) + LP + memoryless_optimal
      ExcessRD                      excess-distortion prior-opt
      TypeBasedBlockLP / …RD        converse-prior LP (block / chord)
      ▲
  fbl  (engines)                    BOUNDS
      OneShot{Channel,RD,JSCC}      exact, lifted |X|^n, + Monte-Carlo
      TypeBased{Channel,RD,JSCC}    method of types, poly-n
      ▲
  fbl  (core, private)              INFRA
      type_class_core               type enumeration / sizes / indexing
      F_curve                       F/A-curve integrators (exact + surrogate kernels)
      channel_achievable_utils      BSC / BEC / Z factories, kronecker_power
      achievable_utils              BMS source, Hamming distortion
      type_based_utils              prior converters (memoryless ↔ type ↔ sequence)
```

## 1. The permutation

| | one-shot | type-based |
|---|---|---|
| **channel** | `OneShotChannel` | `TypeBasedChannel` |
| **rate-distortion** | `OneShotRD` | `TypeBasedRD` |
| **JSCC** | `OneShotJSCC` | `TypeBasedJSCC` |

Each engine exposes: `compute_curve(prior)` → error/distortion spectrum;
`theory(curve, M)` → the bound; `optimize_prior(M)` → the (converse) LP-optimal
prior; and — one-shot only — `draw_random_code` / `evaluate` / `mc` for the
Monte-Carlo cross-check.

## 2. Bounds, converse, and prior optimization

For each setting there are three quantities:

- **achievable** — the random-coding bound (`theory`, or the exact program below);
- **converse** — the meta-converse LP (`optimize_prior` / `compute_converse`);
- **Monte-Carlo** — realised error of drawn codebooks (one-shot only).

The converse already optimises the prior point-wise (one threshold). The
**achievability** bound is a kernel integral of the whole spectrum, so its prior
optimisation is global — and is an **exact convex program**:

| setting | kernel | program | class |
|---|---|---|---|
| channel / JSCC | RCU⁺ (L=1) | exact convex **QP** | `AchievabilityQP`, `AchievabilityJSCC` |
| any positive kernel (incl. RD) | general | **bracketing LP** (secant ≥ truth ≥ tangent) | `*.solve_bracketing_lp` |

The converse is the achievability program with the **Dirac kernel** — used as a
built-in self-check (`AchievabilityJSCC.solve_dirac_ramp` ≡ `compute_converse`).

### 2.1 The unified `Φ`-view and the direct simplex program

Converse and achievability are **the same program**, distinguished only by the
kernel that picks `Φ`. This is the conceptual centre of the library:

![the unified Φ-view](docs/algorithms.png)

<details><summary>Mermaid source</summary>

```mermaid
flowchart TB
  classDef prog fill:#f3e9ff,stroke:#6a3ca6,color:#241040,font-weight:bold;
  classDef conv fill:#e8ecff,stroke:#4456a6,color:#1a2240;
  classDef ach1 fill:#e6f6e6,stroke:#3c7a3c,color:#163016;
  classDef ach2 fill:#ffe9e6,stroke:#a64a3c,color:#3a1410;
  classDef dir  fill:#fff6da,stroke:#9c7a1e,color:#3a2c08,font-weight:bold;
  P["UNIFIED PROGRAM (one simplex optimisation)<br/>minimise &nbsp; P = 1 − cᵀ Φ(A·Q) &nbsp; over &nbsp; Q ∈ simplex<br/>A·Q = cumulative staircase masses σ &nbsp;·&nbsp; c = metric gaps νⱼ−νⱼ₊₁ &nbsp;·&nbsp; kernel chooses Φ (Φ′ = 1−∫κ)"]:::prog
  P --> K1
  P --> K2
  P --> K3
  subgraph CONVERSE[" CONVERSE "]
    direction TB
    K1["kernel = Dirac &nbsp;δ(w−w₀)"]:::conv --> F1["Φ(t) = min(t, w₀) &nbsp;(ramp)"]:::conv --> R1["piecewise-linear ⇒ LP"]:::conv --> S1["meta-converse LP<br/>optimize_prior · compute_converse<br/>TypeBasedBlockLP(RD)"]:::conv
  end
  subgraph ACHA[" ACHIEVABILITY · RCU⁺ (list L=1) "]
    direction TB
    K2["kernel = eᴿ·1{w≤w₀}"]:::ach1 --> F2["Φ(t) = t − ½eᴿt²  (clamped)"]:::ach1 --> R2["piecewise-quadratic ⇒ QP"]:::ach1 --> S2["exact convex QP<br/>AchievabilityQP · AchievabilityJSCC"]:::ach1
  end
  subgraph ACHB[" ACHIEVABILITY · general kernel (RD / excess / list-L) "]
    direction TB
    K3["kernel = general positive"]:::ach2 --> F3["Φ = 1−(1−t)ᴹ  /  degree L+1"]:::ach2 --> R3["secant ≥ truth ≥ tangent ⇒ bracketing LP"]:::ach2 --> S3["AchievabilityLP_RD · ExcessRD<br/>certified gap O(1/K²)"]:::ach2
  end
  S1 -.-> DIR
  S2 -.-> DIR
  S3 -.-> DIR
  DIR["DirectPriorOpt — exact for ANY kernel<br/>first-order water-fill march on the simplex<br/>optimality: g(x)=∂Γ/∂Q equal on the support"]:::dir
```
</details>

All of the above are one program, `minimize 1 − cᵀ Φ(A·Q)` over the simplex, where
`A·Q` are the cumulative staircase masses, `c` the metric gaps, and the **kernel
chooses `Φ`** (`Φ' = 1 − g`, `g = ∫κ`):

| program | kernel | `Φ(t)` | shape |
|---|---|---|---|
| converse | Dirac `δ(w−w₀)` | `min(t, w₀)` (ramp) | piecewise-linear → **LP** |
| RCU⁺ (L=1) | `e^R 1{w≤w₀}` | `t − ½e^R t²` clamped | piecewise-quadratic → **QP** |
| list-L / RD | — | degree `L+1` / `1−(1−t)^M` | → LP / bracket |

`DirectPriorOpt` solves this **directly on the simplex** by a first-order march
(projected gradient / Frank–Wolfe) using the analytic water-fill gradient
`g(x) = ∂Γ/∂Q(x)`; the directional derivative on the simplex is `⟨g − ḡ, μ⟩`
(`Σμ = 0`), and optimality is the **water-filling** condition (`g(x)` equal on the
support). It is exact for *any* kernel (no bracketing gap), cheap per-iteration
(no cvxpy), and warm-startable across a rate sweep.

**Solver choice (recommended).**

| task | default | `DirectPriorOpt` |
|---|---|---|
| achievability, RCU⁺ | **exact QP** | alternative; scalable + warm-startable, first-order (slow to ~1e-9) |
| achievability, general kernel (RD/JSCC list-L) | bracketing LP (`O(h²)`) | **exact** (true `Φ`, no bracket gap) |
| converse | **meta-converse LP** (exact, one-shot) | unified view only; non-smooth (Dirac) → first-order crawls |

The direct program is the *unifying* lens and the scalable/exact-for-any-kernel
path; the QP/LP remain the robust exact solvers (interior-point handles the
flat/degenerate optimum that stalls plain first-order methods). A
support-identifying active-set finish (future work) would make the direct program
competitive at machine precision.

## 3. Conventions (read before comparing curves)

- **Rate / `M`.** Channel has a free rate knob `M = e^{nR}` (per-symbol
  `M₁ = e^R`). JSCC has **no free knob**: you resolve the whole source, so for
  list size `L=1` the codebook is pinned to `M = |V|^n` (rate-lowering levers are
  the list size and the source entropy). RD uses `M` reproduction candidates.
- **Kernel consistency.** The QP optimises the **RCU⁺** bound; compare it against a
  memoryless prior scored with the **same** RCU⁺ kernel
  (`fbl.prioropt.rcu_plus_from_F_curve`, threshold `1/M`), **not** the exact-RC
  `theory` (a different, tighter bound). Mixing kernels breaks the
  `optimal ≤ memoryless` ordering.
- **Standard memoryless baseline — `memoryless_optimal(M)`.** Defined as *the
  optimal single-letter prior (exact n=1 program) applied i.i.d.* It is fast
  (constant in `n`), exact, and the i.i.d. extension is a feasible point of the
  n-letter program, so `QP@n ≤ memoryless_optimal@n` holds **by construction**.

## 4. Validation strategy

No closed-form oracle exists in general, so every bound is cross-checked
(`tests/`):

1. **one-shot ↔ type-based** agreement at small `n` (F/A-curves and the LP optimum);
2. **RCU expectation ↔ Monte-Carlo** mean of random codebooks;
3. **converse ≤ achievable** at every rate;
4. **prior-opt program ↔ invariants**: exact QP ≤ best memoryless; bracketing LP
   straddles the exact bound at its own optimum (`P_lo ≤ P ≤ P_hi`); Dirac program
   ≡ meta-converse LP.

## 5. Known ceilings

- **Type-based** is polynomial in `n` but the **cvxpy build** of the QP/LP grows
  with the staircase row count; practical to `n ≈ 12–20` before constraint
  assembly (not the solve) dominates. Vectorising the per-knot constraint build is
  the route to larger `n`.
- **Bracketing LP** mesh `K` trades the certified gap (`O(1/K²)`) against build
  cost; `K ≈ 20–48` is plenty.
- **One-shot / Monte-Carlo** is exponential in `n` (`|X|^n`); practical to
  `n ≈ 10–12`.

## 6. A note on the headline findings

- **Channel.** The achievability prior gain over optimised memoryless is a
  **low-rate + large-`n` corner** effect (constant-composition): small at `n=6`,
  large (tens of %) by `n=20` at low rate.
- **JSCC.** For an i.i.d. source + memoryless channel the gain is **negligible**
  at small `n` — the optimal ensemble is essentially memoryless. See
  `examples/ex5_jscc_gain.py`.
