# Results

Each use case is pinned to a single channel/source and shown as four figures —
**G1** bound vs Monte-Carlo (validation), **G2** the **prior gap** (the
centerpiece: optimal achievable prior vs optimal memoryless vs the two
*marginal-memoryless* priors), **G3** exact random coding vs a closed-form
surrogate, **G4** the error/distortion spectrum of the converse- vs
achievability-optimal prior.

The achievability-optimal prior is computed by the **Φ-view simplex march**
(KKT / Frank–Wolfe-gap certified — an intrinsic optimality proof), scored with the
**exact** random-coding kernel. G1 validates the bound at small `n`; the result
figures (G2–G4) use the type-based representation to reach **`n=20`** (excess stays
at `n=6` — it is a lifted block quantity).

| Use case | Pinned case | Results | Generator |
|---|---|---|---|
| Channel coding | Z(0.1), `n=8,20` | [results/channel.md](results/channel.md) | `examples/gen_channel.py` |
| RD — average | BMS(0.25)+Hamming, `n=8,20` | [results/rd_average.md](results/rd_average.md) | `examples/gen_rd_average.py` |
| RD — excess | BMS(0.25), `T=1`, `n=6` | [results/rd_excess.md](results/rd_excess.md) | `examples/gen_rd_excess.py` |
| JSCC | (future work) | [results/jscc.md](results/jscc.md) | `examples/gen_jscc.py` |

Every bound is backed by the cross-check suite in [`tests/`](tests/) (**106 tests**:
engine cross-checks + Φ-view identity + simplex-march optimality/KKT). See
[`docs/TESTING.md`](docs/TESTING.md) for what each check guarantees.

```bash
pip install -e ".[plots,test]"
python examples/generate_all.py     # -> examples/figures/*.png
pytest -q                           # 106 tests
```

## Headline across use cases

The reframed question is: **we have the exact, certified optimal prior — how big is
the gap to memoryless?** The answer separates the settings:

| use case | G2 gain (optimal vs best memoryless) | marginal recovers it? | G4 penalty (converse prior reused) |
|---|---|---|---|
| channel Z(0.1) | 2.0 % (`n=8`) → 3.3 % (`n=20`) | yes (achiev. marginal); converse marginal lags at low rate | **5× → 15×** (grows with `n`) |
| RD average | ~5.4 % (`n=8`) → 5.9 % (`n=20`), high-rate | yes (both marginals ≈ optimal) | ~1.4 % (priors nearly identical) |
| RD excess | up to ~7 % | achiev. marginal yes; **converse marginal poor** | **2.8×** |

Two robust conclusions, now stated *rigorously* (the optimum is KKT-certified, not
heuristic): (1) for these settings the memoryless prior is **nearly optimal** — the
non-product gain is a few percent, growing slowly with `n`; (2) the converse prior
is a **bad achievability prior** (G4), and *marginalizing* it mostly fixes that for
channel but not for excess. See [`docs/THEORY.md`](docs/THEORY.md) §7.1 on
marginalization and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the kernel-consistency
convention that makes these comparisons apples-to-apples.
