# Results

Headline figures and numbers produced by [`examples/generate_all.py`](examples/)
(reduced, fast settings — representative, not thesis-resolution). Every bound is
backed by the cross-check suite in [`tests/`](tests/) (**64 tests passing**:
52 engine cross-checks + 12 prior-optimization invariants).

Regenerate everything with:

```bash
pip install -e ".[plots,test]"
python examples/generate_all.py     # -> examples/figures/*.png
pytest -q                           # 64 tests
```

---

## G1 — Monte-Carlo spread around the RCU expectation (channel)

![MC spread](examples/figures/ex1_mc_spread.png)

The achievable bound is an *expectation* over the random-coding ensemble. 60
actual codebooks (drawn in the lifted `X^6` space, exact ML decoding) scatter
around the analytic RCU expectation — a single random code can deviate
materially from the mean at finite `n`. *(Z(0.1), n=6.)*

---

## G2 — the prior gap, with the true achievability optimum (channel)

![prior gap](examples/figures/ex2_prior_gap.png)

The distinguishing figure: the **exact achievability-optimal prior** (QP, over all
type priors) vs the **best memoryless prior**, against the meta-converse, at
`n=12`. The QP gain over the best memoryless prior peaks at **≈2.7 % at low rate**
and decays to ~0 at high rate — the **low-rate corner** of the constant-composition
effect. (At `n=20` this corner reaches tens of %; here `n` is reduced for speed.)
The QP is, as it must be, never below the meta-converse.

---

## G3 — exact random-coding bound vs the closed-form surrogate (channel & RD)

![bounds vs exact](examples/figures/ex3_bounds_vs_exact.png)

The exact random-coding kernel vs the common closed-form replacements — the
**union bound** (channel) and the **exponential bound** (rate-distortion). Both
surrogates are loose at low rate and tighten as the rate grows. *(n=10.)*

---

## F-curve comparison — converse-optimal vs achievability-optimal prior

![F-curve compare](examples/figures/ex4_fcurve_compare.png)

The converse and achievability bounds optimise the prior for *different*
objectives, so their optimal priors — and their error spectra — genuinely differ.
The achievability-optimal prior concentrates spectral mass below the kernel
threshold `w_0 = 1/M`; the converse-optimal prior (tuned for the single
meta-converse threshold) has a more gradual spectrum. *(Z(0.1), n=12, R=0.25 bits;
the two priors' bound values, `5.2e-6` converse vs `1.0e-2` achievability, are
different bound types — a converse lower bound and an achievability upper bound —
illustrating the finite-`n` converse–achievability gap.)*

---

## JSCC — the non-product prior gain is tiny

![JSCC gain](examples/figures/ex5_jscc_gain.png)

For an i.i.d. source through a memoryless channel, the exact full-type-prior QP
optimum buys almost nothing over the **best memoryless prior at blocklength `n`**:
the genuine non-product gain (red) stays **well under 1 % through n=5**. The larger
gain vs the *single-letter-optimal* prior applied i.i.d. (blue, ≈3 % at n=5) is
mostly a **within-memoryless** effect — the n=1-optimal prior is itself suboptimal
as a memoryless prior at larger `n` — not a non-product effect. This is the JSCC
counterpart of the channel's G2 result, and it is essentially null: the source
structure already lives in the metric, so a memoryless conditional law captures it.

| `n` | QP | best memoryless@`n` | non-product gain |
|----:|----:|----:|----:|
| 2 | 0.37735 | 0.37738 | 0.008 % |
| 3 | 0.36581 | 0.36640 | 0.162 % |
| 4 | 0.36026 | 0.36123 | 0.269 % |
| 5 | 0.35994 | 0.36114 | 0.333 % |

---

## Validation summary

| check | where | result |
|---|---|---|
| one-shot ↔ type-based (F/A-curves, LP) | `tests/test_type_based_*` | exact to fp |
| RCU expectation ↔ Monte-Carlo mean | `tests/test_jscc_one_shot`, `ex1` | agree |
| converse ≤ achievable | `tests/test_prioropt` | holds |
| exact QP ≤ best memoryless (RCU⁺ kernel) | `tests/test_prioropt` | holds |
| bracketing LP straddles exact (`P_lo ≤ P ≤ P_hi`) | `tests/test_prioropt` | holds |
| Dirac-kernel program ≡ meta-converse LP | `tests/test_prioropt` | ~1e-9 |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the conventions (rate/`M` semantics,
RCU⁺-vs-exact kernel, the `memoryless_optimal` baseline) that make these
comparisons apples-to-apples.
