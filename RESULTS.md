# Results

Finite-blocklength figures, one set per use case. Each set has four figures —
**G1** Monte-Carlo spread vs the achievable expectation, **G2** the prior gap
(achievability-optimal vs best memoryless), **G3** exact random coding vs a
closed-form surrogate, **G4** the error/distortion spectrum of the
converse-optimal vs the achievability-optimal prior.

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
