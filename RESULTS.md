# Results

Finite-blocklength figures, one set per use case. Each set has up to five
figures — **G1** Monte-Carlo spread vs the achievable expectation, **G2** the
prior gap (achievability-optimal vs best memoryless), **G3** exact random coding
vs a closed-form surrogate, **G4** the error/distortion spectrum of the
converse-optimal vs the achievability-optimal prior, and **G5** the
*marginalization* test — each optimal prior vs its i.i.d. per-symbol marginal
(the classical error-exponent recipe for a memoryless prior).

| Use case | Results | Generator |
|---|---|---|
| Channel coding | [results/channel.md](results/channel.md) | `examples/gen_channel.py` |
| Rate-distortion — average | [results/rd_average.md](results/rd_average.md) | `examples/gen_rd_average.py` |
| Rate-distortion — excess | [results/rd_excess.md](results/rd_excess.md) | `examples/gen_rd_excess.py` |
| JSCC | [results/jscc.md](results/jscc.md) | `examples/gen_jscc.py` |

Reduced, fast settings (representative, not thesis-resolution). Every bound is
backed by the cross-check suite in [`tests/`](tests/) (**68 tests passing**:
52 engine cross-checks + 16 prior-optimization invariants). See
[`docs/TESTING.md`](docs/TESTING.md) for what each check guarantees.

```bash
pip install -e ".[plots,test]"
python examples/generate_all.py     # -> examples/figures/*.png
pytest -q                           # 68 tests (~25 min; target a file to iterate)
```

## Headline across use cases

The **G4** comparison — how badly the converse-optimal prior does when reused for
achievability — separates the settings:

| use case | G2 gain (optimal vs memoryless) | G4 penalty (converse prior reused for achievability) |
|---|---|---|
| channel Z(0.1) | ~2.7 % (low-rate corner) | **8.5×** worse |
| RD average | 1.4–3.75 % | ~1 % (priors nearly identical) |
| RD excess | 3.5–7 % | **2.8×** worse |
| JSCC (i.i.d. src) | <1 % (null) | — |

Excess distortion and channel coding are strongly prior-sensitive; average
distortion is not; JSCC's non-product gain is essentially nil. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the conventions that make these
comparisons apples-to-apples.

## Marginalization (G5): is the per-symbol marginal a good memoryless prior?

The classical error-exponent recipe for a memoryless prior is to take the
**per-symbol marginal** of a general prior and apply it i.i.d. G5 tests it: each
optimal prior's F-curve vs its marginalized i.i.d. version.

**The achievable bound under each prior** (full vs marginalized), reproducible via
[`examples/marginalize_table.py`](../examples/marginalize_table.py):

| use case | metric | achiev-opt (full) | achiev-opt (marginal) | converse (full) | converse (marginal) |
|---|---|---|---|---|---|
| channel Z(0.1) | `P_e` | **0.010** | 0.0103 | 0.0847 | 0.0113 |
| RD average | `D`/sym | **0.153** | 0.160 | 0.155 | 0.158 |
| RD excess | `P_exc` | **0.0162** | 0.0170 | 0.0451 | 0.0474 |

Reading across a row answers the question directly. **Marginalizing the
achievability-optimal prior** barely moves the bound (+2.7 % / +4.4 % / +4.9 %):
the marginal is essentially as good as the full type-prior optimum, because at
these `n` the optimum is already nearly i.i.d. **Marginalizing the converse
prior** is dramatic for the channel — `0.085 → 0.011`, an ~8× *improvement* that
lands it right next to the achievability optimum — and modest for RD (where the
two priors were close to begin with). The marginal discards exactly the
non-product, single-threshold structure that the achievability integral
penalizes, so the rescue is largest where that structure mattered most.

(JSCC is omitted: its prior is a *conditional* law `Q_{X|V}` and its non-product
gain is already ~null, so marginalization is a no-op of interest there.)
