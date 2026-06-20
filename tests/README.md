# Tests

There is **no closed-form oracle** for finite-blocklength bounds, so the suite
validates by **cross-checking** quantities computed by independent routes (and by
*intrinsic* certificates). **107 tests.**

```bash
pytest -q                              # whole suite (slow: cvxpy + lifted |X|^n)
pytest tests/test_phi_view.py -q       # one file
pytest -q -k "kkt or march"            # by keyword
```

Use `C:\Users\User\anaconda3\python.exe -m pytest` on this machine if `python`
isn't on PATH. All randomness is seeded — the suite is not flaky.

## What each file covers

| file | tests | what it guarantees |
|---|---|---|
| `test_one_shot.py` | channel/RD one-shot run | exercises `compute_curve`/`mc`/`optimize_prior`/`check_kkt`. **Note: print-only (no asserts)** — see "gaps" below. |
| `test_type_based_channel.py` | F-curve & LP suites | **one-shot ↔ type-based** agreement (channel) at small `n`; LP optimum matches at `n=1`, `≤` at `n>1`. |
| `test_type_based_rd.py` | A-curve & LP suites | same, rate-distortion. |
| `test_jscc_one_shot.py` | BSC / Z / random / KKT / evaluate | **RCU ↔ Monte-Carlo** (3σ sandwich), converse ≤ achievable, KKT asserted. |
| `test_jscc_type_based.py` | n=1,2,3 suites | JSCC one-shot ↔ type-based (curves, converse, achievable); memoryless-prior bridge. |
| `test_prioropt.py` | QP / bracket invariants | exact QP ≤ best memoryless; bracketing LP straddles the exact bound; JSCC Dirac ≡ meta-converse; converse ≤ achievable; excess bracket straddles. |
| `test_phi_view.py` | the relaxation `J = cᵀΦ(A·Q)` | **formula == direct integral** (channel/RD exact+upper); **type-based == lifted one-shot** (channel/RD/JSCC); RD-exact tied to the MC-validated one-shot. |
| `test_phi_simplex.py` | the simplex march | march optimum **== exact QP** (channel, JSCC) and **inside the bracket** (RD); analytic gradient == finite difference; **KKT certificate** holds at the optimum and is **rejected** off it (negative control); JSCC per-block KKT. |

## The validation layers

1. **one-shot ↔ type-based** at small `n` (`test_type_based_*`, `test_jscc_type_based`).
2. **RCU expectation ↔ Monte-Carlo** (`test_jscc_one_shot`; `test_phi_view` ties
   the Φ-view to the MC-validated one-shot path).
3. **converse ≤ achievable** at every rate (`test_jscc_*`, `test_prioropt`).
4. **the relaxation is exact**: `J = cᵀΦ(A·Q)` equals an independent computation,
   in both representations and all settings (`test_phi_view`).
5. **the optimum is optimal**: the march matches the exact solvers *and* satisfies
   the intrinsic **KKT water-filling certificate** (`test_phi_simplex`).

See [`../docs/TESTING.md`](../docs/TESTING.md) for the detailed guarantee of each
check and the tolerance rationale.

## Known gaps (where a bug could slip through)

- **`test_one_shot.py` is print-only** (no `assert`): it runs the channel/RD
  one-shot MC and KKT paths but does not *enforce* them, so strategies (2)/(3) are
  asserted only for JSCC there. Adding `assert res["within_ci"]` and
  `assert kkt["cond1"] and kkt["cond2"]` would close this.
- The `build_A_curve_type_based` **A-curve integrator is wrong at `n=1`** for
  non-uniform priors (affects `exact_D_rand`/`true_D_at_Q` only; the `cᵀΦ`
  staircase path used by the march is correct at all `n`, and the figures use
  `n ≥ 6`). Surfaced by `test_phi_view`'s one-shot tie; not yet fixed.
