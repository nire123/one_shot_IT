# Testing & validation

There is **no closed-form oracle** for finite-blocklength bounds in general, so
the suite validates by **cross-checking** quantities that must agree (or must be
ordered) but are computed by independent paths. This doc maps each test file onto
the guarantee it provides.

## Running

```bash
pip install -e ".[plots,test]"
pytest -q
```

The suite is **107 tests**. It is **slow** (tens of minutes): the prior-opt and
one-shot tests build cvxpy programs and lift to `|X|ⁿ` at `n` up to 6–8. For a fast
inner loop, target a file or a setting (see also
[`../tests/README.md`](../tests/README.md) for a per-file map):

```bash
pytest tests/test_prioropt.py -q            # prior-opt invariants only
pytest tests/test_type_based_channel.py -q  # one engine
pytest -q -k "qp or bracket"                # by name
```

All randomness is seeded (`np.random.default_rng(...)`, fixed Monte-Carlo seeds),
so the suite is **not flaky**; there are no `skip`/`xfail` markers.

## The validation strategies

| # | cross-check | guarantees | where |
|---|---|---|---|
| 1 | **one-shot ↔ type-based** at small `n` | the method-of-types reconstruction equals the exact lifted computation (F/A-curves and the converse LP optimum) | `test_type_based_channel.py`, `test_type_based_rd.py`, `test_jscc_type_based.py` |
| 2 | **RCU expectation ↔ Monte-Carlo** | the analytic achievable bound equals the mean realised error of drawn random codebooks (3σ sandwich) | `test_jscc_one_shot.py` |
| 3 | **converse ≤ achievable** | the meta-converse never exceeds the achievable bound at any rate | `test_jscc_one_shot.py`, `test_jscc_type_based.py`, `test_prioropt.py` |
| 4 | **prior-opt invariants** | the exact QP/bracket programs satisfy their structural guarantees | `test_prioropt.py` |
| 5 | **Φ-view identity** | the relaxation `J = cᵀΦ(A·Q)` equals an independent computation (formula vs direct, type-based vs lifted) for channel/RD/JSCC | `test_phi_view.py` |
| 6 | **march optimality** | the simplex march matches the exact solvers and satisfies the intrinsic **KKT** certificate (rejected off the optimum) | `test_phi_simplex.py` |

### Strategy 4 in detail (the contribution's checks)

- **`QP ≤ best memoryless`** — the exact achievability QP can never be worse than
  the optimal memoryless prior (the latter is a feasible point). *Channel:*
  `test_prioropt.py::test_channel_qp_le_best_memoryless`; *JSCC:*
  `::test_jscc_qp_le_memoryless_optimal`.
- **Bracket straddles the truth** — `P_lo ≤ exact ≤ P_hi` (and similarly for `D`,
  `P_exc`) at the bracket's own optimum.
  `::test_channel_bracket_contains_exact`, `::test_rd_bracket_straddles_exact`,
  `::test_excess_bracket_straddles_exact`.
- **Dirac program ≡ meta-converse LP** — running the achievability skeleton with
  the ramp kernel reproduces the converse value.
  `::test_jscc_dirac_equals_converse`.
- **Unified-view consistency** — the `DirectPriorOpt` water-fill gradient matches
  a finite-difference directional derivative, and its RCU⁺ optimum matches the QP;
  the converse (Dirac) program is zero below capacity.
  `test_direct_program.py`.

## Tolerances — how to read a pass

- **Exact equalities** (one-shot ↔ type-based at `n=1`; Dirac ≡ converse) use
  `< 1e-5` — same search space, machine-exact match expected.
- **Inequalities** (`QP ≤ memoryless`, `converse ≤ achievable`) carry a small
  one-sided slack (`±1e-6`/`1e-5`) for solver noise; the slack is **directional**,
  so a genuine violation in the wrong direction still fails.
- **Monte-Carlo** uses a `3σ` band (~99.7%) with fixed seeds — statistically
  sound and reproducible.

## Coverage notes

For readers extending the suite, the current gaps (a bug here could pass
undetected) are:

- `tests/test_one_shot.py` exercises the channel/RD one-shot Monte-Carlo and KKT
  paths but is **print-only** (no asserts): it runs the code and would surface a
  crash, but does not *enforce* strategies (2)/(3) for the plain channel/RD
  engines — those are asserted only for JSCC. Adding `assert res["within_ci"]`
  and `assert kkt["cond1"] and kkt["cond2"]` would close this.
- Excess distortion is asserted only at threshold `d_th = 0`.
- Type-based converse at `n > 1` uses a looser `1e-3` tolerance than the `n = 1`
  `1e-4`.

These do not affect the validity of the bounds that *are* checked (strategies 1–4
above all pass); they mark where the safety net is thinner.
