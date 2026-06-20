"""
Rate-distortion (average distortion) -- results for the pinned case
BMS(p=0.25) + Hamming.

  G1  bound vs Monte-Carlo            (validation; small n only)
  G2  prior gap: optimal achievable prior vs optimal memoryless vs
      marginal-memoryless (achievable / converse)            -- the centerpiece
  G3  exact random-coding distortion vs the exponential surrogate
  G4  distortion spectrum: achievability- vs converse-optimal prior

The achievability-optimal reproduction prior is the Phi-view simplex march
(exact kernel  Phi(tau)=(1-tau)^M, M=e^{nR}; D = c^T Phi(A Q)).  G1 stays at
n=8; G2/G3/G4 run at n=8 and n=20.

Run:  python examples/gen_rd_average.py  ->  examples/figures/rd_avg_*.png
"""
import numpy as np
from _common import save, plt

from fbl import OneShotRD, TypeBasedRD
from fbl.achievable_utils import setup_bms_hamming
from fbl.type_based_utils import memoryless_to_type_prior
from fbl.F_curve import integrate_curve_rd_exact, integrate_curve_rd_exp_bound
from fbl.prioropt.typebased_block_lp import marginal_input
from fbl.prioropt import phi_simplex as ps
from fbl.prioropt import phi_view as pv

P = 0.25
P_X1 = np.array([1 - P, P])
D_SINGLE = np.array([[0.0, 1.0], [1.0, 0.0]])
KY = D_SINGLE.shape[1]
PREFIX = "rd_avg"
TITLE = "RD avg. (BMS p=0.25 + Hamming)"
LN2 = np.log(2.0)


def _d_exact(n, Q_type, M):
    """Per-symbol distortion of a reproduction type prior, exact best-of-M kernel."""
    return pv.J_typebased_rd(P_X1, D_SINGLE, n, Q_type, M, "exact") / n


def _best_memoryless(n, M, grid):
    return min(_d_exact(n, memoryless_to_type_prior(np.array([1 - q, q]), n), M)
               for q in grid)


# ── G1 ─ bound vs Monte-Carlo (validation, n=8) ──────────────────────────────
def g1_mc_spread(n=8):
    P_X, d, _ = setup_bms_hamming(P, n)
    osr = OneShotRD(P_X, d)
    Y = d.shape[1]
    Q = np.ones(Y) / Y
    curve = osr.compute_curve(Q)
    R_bits = np.linspace(0.08, 0.85, 12)
    Ms = [max(2, int(round(np.exp(n * Rb * LN2)))) for Rb in R_bits]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _ in range(60):
        dd = [osr.evaluate(osr.draw_random_code(Q, M, rng)) / n for M in Ms]
        ax.plot(R_bits, dd, color="C0", alpha=0.10, lw=0.8)
    ax.plot([], [], color="C0", alpha=0.4, label="60 random codebooks")
    ax.plot(R_bits, [osr.theory(curve, M) / n for M in Ms], "k-", lw=2.2,
            label="RC expectation (exact)")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel("distortion / symbol")
    ax.set_title(f"G1  {TITLE}, $n={n}$: bound vs Monte-Carlo")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return save(fig, f"{PREFIX}_g1_mc_spread.png")


# ── G2 ─ prior gap (exact kernel) ────────────────────────────────────────────
def g2_prior_gap(n):
    tbr = TypeBasedRD(P_X1, D_SINGLE, n)
    prog = ps.build_program("rd", P_X=P_X1, d=D_SINGLE, n=n, kernel="exact")
    R_bits = np.linspace(0.15, 0.8, 12)
    mgrid = np.linspace(0.02, 0.98, 33)

    d_ach, d_ml, d_am, d_cm, d_conv = [], [], [], [], []
    warm = None
    for Rb in R_bits:
        M = float(np.exp(n * Rb * LN2))
        res = ps.optimize(prog, M, method="pgd", max_iter=3000, tol=1e-8, warm_start=warm)
        warm = res["Q"]
        Q_ach = res["Q"]
        Q_conv, _ = tbr.optimize_prior(M)
        Q_am = memoryless_to_type_prior(marginal_input(Q_ach, n, KY), n)
        Q_cm = memoryless_to_type_prior(marginal_input(Q_conv, n, KY), n)
        d_ach.append(res["J"] / n)
        d_ml.append(_best_memoryless(n, M, mgrid))
        d_am.append(_d_exact(n, Q_am, M))
        d_cm.append(_d_exact(n, Q_cm, M))
        d_conv.append(tbr.optimize_prior(M)[1] / n)
    A = lambda v: np.array(v)
    d_ach, d_ml, d_am, d_cm, d_conv = map(A, (d_ach, d_ml, d_am, d_cm, d_conv))
    gain = (d_ml - d_ach) / d_ml * 100.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax.plot(R_bits, d_ml, "o-", color="C0", label="optimal memoryless")
    ax.plot(R_bits, d_am, "v--", color="C2", label="marginal memoryless (achiev.)")
    ax.plot(R_bits, d_cm, "^:", color="C3", label="marginal memoryless (converse)")
    ax.plot(R_bits, d_ach, "s-", color="C1", label="optimal achievable prior")
    ax.plot(R_bits, d_conv, "k-", lw=1, alpha=0.6, label="single-threshold (converse)")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel("distortion / symbol")
    ax.set_title(f"G2  {TITLE}, $n={n}$: prior gap"); ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax2.plot(R_bits, gain, "s-", color="C1")
    ax2.set_xlabel("rate $R$ (bits/sym)")
    ax2.set_ylabel("gain of optimal over best memoryless (%)")
    ax2.set_title("non-product prior gain"); ax2.grid(True, alpha=0.3)
    print(f"  G2 n={n}: max gain {gain.max():.2f}% at R={R_bits[gain.argmax()]:.2f}b")
    return save(fig, f"{PREFIX}_g2_prior_gap_n{n}.png")


# ── G3 ─ exact RC vs exponential surrogate ───────────────────────────────────
def g3_bounds_vs_exact(n):
    tbr = TypeBasedRD(P_X1, D_SINGLE, n)
    curve = tbr.compute_curve(memoryless_to_type_prior(np.array([0.5, 0.5]), n))
    R_bits = np.linspace(0.1, 0.85, 16)
    Ms = [np.exp(n * Rb * LN2) for Rb in R_bits]
    ex = [integrate_curve_rd_exact(*curve, M) / n for M in Ms]
    eb = [integrate_curve_rd_exp_bound(*curve, M) / n for M in Ms]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(R_bits, ex, "o-", label="exact RC")
    ax.plot(R_bits, eb, "s--", label="exponential bound")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel("distortion / symbol")
    ax.set_title(f"G3  {TITLE}, $n={n}$: bound vs exact RC")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return save(fig, f"{PREFIX}_g3_bounds_vs_exact_n{n}.png")


# ── G4 ─ distortion spectrum: achievability- vs converse-optimal prior ───────
def g4_fcurve_compare(n):
    Rb = 0.4
    M = float(np.exp(n * Rb * LN2)); w0 = 1.0 / M
    tbr = TypeBasedRD(P_X1, D_SINGLE, n)
    prog = ps.build_program("rd", P_X=P_X1, d=D_SINGLE, n=n, kernel="exact")
    P_st, _ = tbr.optimize_prior(M)
    P_ach = ps.optimize(prog, M, method="pgd", max_iter=4000, tol=1e-8)["Q"]
    d_st = _d_exact(n, P_st, M)
    d_ach = _d_exact(n, P_ach, M)

    ks, As = tbr.compute_curve(P_st)
    ka, Aa = tbr.compute_curve(P_ach)
    w = np.linspace(0, min(1.0, 8 * w0), 400)

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ax.axvspan(0, w0, color="0.92", label=r"best-of-$M$ weight ($w\lesssim 1/M$)")
    ax.plot(w, np.interp(w, ks, As) / n, color="C0", lw=2,
            label=f"converse-optimal prior  (D={d_st:.3f})")
    ax.plot(w, np.interp(w, ka, Aa) / n, color="C1", lw=2,
            label=f"achievability-optimal prior  (D={d_ach:.3f})")
    ax.axvline(w0, color="k", ls=":", lw=1.2)
    ax.set_xlabel("reproduction mass $w$")
    ax.set_ylabel("distortion spectrum $A(w)$ / symbol")
    ax.set_title(f"G4  {TITLE}, $n={n}$, $R={Rb}$: achievable vs converse prior")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
    print(f"  G4 n={n}: converse D={d_st:.4f}, achievable D={d_ach:.4f}")
    return save(fig, f"{PREFIX}_g4_fcurve_compare_n{n}.png")


def main():
    print("[g1]"); g1_mc_spread(8)
    for n in (8, 20):
        print(f"[g2 n={n}]"); g2_prior_gap(n)
        print(f"[g3 n={n}]"); g3_bounds_vs_exact(n)
        print(f"[g4 n={n}]"); g4_fcurve_compare(n)


if __name__ == "__main__":
    main()
