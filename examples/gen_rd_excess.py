"""
Rate-distortion (EXCESS distortion) -- results for the pinned case
BMS(p=0.25), block-Hamming, threshold T=1, n=6.

Excess distortion is best-of-M of the block indicator d_exc = 1{Hamming > T}:
    P_exc(Q) = sum_x P_X(x) (1 - q(x))^M,   q(x) = Q{y : d(x,y) <= T}.
This is a lifted Y^n quantity (the per-letter indicator is degenerate), so all
figures stay at n=6 (the rare-event Monte-Carlo for G1 also needs small n).

  G1  bound vs Monte-Carlo
  G2  prior gap: optimal achievable prior vs optimal memoryless vs
      marginal-memoryless (achievable / converse)            -- the centerpiece
  G3  exact P_exc vs the exponential surrogate
  G4  excess spectrum: achievability- vs converse-optimal prior

Run:  python examples/gen_rd_excess.py  ->  examples/figures/rd_exc_*.png
"""
import numpy as np
import cvxpy as cp
from _common import save, plt

from fbl import OneShotRD
from fbl.achievable_utils import binary_memoryless_source, hamming_distortion
from fbl.F_curve import integrate_curve_rd_exp_bound

P = 0.25
N = 6
T = 1
PREFIX = "rd_exc"
TITLE = f"RD excess (BMS p={P}, $n={N}$, $T={T}$)"
LN2 = np.log(2.0)


def _setup():
    P_X = binary_memoryless_source(P, N)
    H = hamming_distortion(N)
    d_exc = (H > T).astype(float)
    return P_X, d_exc


def _Pexc_memoryless(P_X, cover, q1, M):
    n = int(round(np.log2(len(P_X))))
    Qy = np.array([q1[1] ** bin(y).count("1") * q1[0] ** (n - bin(y).count("1"))
                   for y in range(len(P_X))])
    Qy /= Qy.sum()
    return float(np.sum(P_X * (1 - cover @ Qy) ** M))


def _Pexc_opt(P_X, cover, M):
    """Achievability-optimal reproduction prior over Y^n (convex)."""
    Y = cover.shape[1]
    Q = cp.Variable(Y, nonneg=True)
    q = cover @ Q
    obj = cp.sum(cp.multiply(P_X, cp.power(1 - q, M)))
    cp.Problem(cp.Minimize(obj), [cp.sum(Q) == 1]).solve(solver=cp.CLARABEL)
    return float(obj.value), np.asarray(Q.value, float)


def _marginal_iid_lifted(Q, n):
    ones = np.array([bin(y).count("1") for y in range(len(Q))])
    q1 = float((Q * ones).sum() / n)
    Qm = np.array([q1 ** c * (1 - q1) ** (n - c) for c in ones])
    return Qm / Qm.sum()


def _pexc(P_X, cover, Q, M):
    return float(np.sum(P_X * (1 - cover @ Q) ** M))


# ── G1 ───────────────────────────────────────────────────────────────────────
def g1_mc_spread():
    P_X, d_exc = _setup()
    osr = OneShotRD(P_X, d_exc)
    Y = d_exc.shape[1]
    Q = np.ones(Y) / Y
    curve = osr.compute_curve(Q)
    R_bits = np.linspace(0.25, 0.85, 12)
    Ms = [max(2, int(round(np.exp(N * Rb * LN2)))) for Rb in R_bits]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _ in range(60):
        pe = [max(osr.evaluate(osr.draw_random_code(Q, M, rng)), 1e-6) for M in Ms]
        ax.semilogy(R_bits, pe, color="C0", alpha=0.10, lw=0.8)
    ax.plot([], [], color="C0", alpha=0.4, label="60 random codebooks")
    ax.semilogy(R_bits, [osr.theory(curve, M) for M in Ms], "k-", lw=2.2,
                label="expectation (exact)")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel(r"excess prob. $P_{exc}$")
    ax.set_title(f"G1  {TITLE}: bound vs Monte-Carlo")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    return save(fig, f"{PREFIX}_g1_mc_spread.png")


# ── G2 ─ prior gap ───────────────────────────────────────────────────────────
def g2_prior_gap():
    P_X, d_exc = _setup()
    cover = (d_exc == 0).astype(float)
    osr = OneShotRD(P_X, d_exc)
    qgrid = np.linspace(0.02, 0.98, 33)
    R_bits = np.linspace(0.3, 0.95, 9)

    pe_ach, pe_ml, pe_am, pe_cm, pe_conv = [], [], [], [], []
    for Rb in R_bits:
        M = float(np.exp(N * Rb * LN2))
        _, Q_ach = _Pexc_opt(P_X, cover, M)
        Q_conv, st = osr.optimize_prior(M)
        pe_ach.append(_pexc(P_X, cover, Q_ach, M))
        pe_ml.append(min(_Pexc_memoryless(P_X, cover, np.array([1 - q, q]), M) for q in qgrid))
        pe_am.append(_pexc(P_X, cover, _marginal_iid_lifted(Q_ach, N), M))
        pe_cm.append(_pexc(P_X, cover, _marginal_iid_lifted(Q_conv, N), M))
        pe_conv.append(st)
    A = lambda v: np.array(v)
    pe_ach, pe_ml, pe_am, pe_cm, pe_conv = map(A, (pe_ach, pe_ml, pe_am, pe_cm, pe_conv))
    gain = (pe_ml - pe_ach) / pe_ml * 100.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax.semilogy(R_bits, pe_ml, "o-", color="C0", label="optimal memoryless")
    ax.semilogy(R_bits, pe_am, "v--", color="C2", label="marginal memoryless (achiev.)")
    ax.semilogy(R_bits, pe_cm, "^:", color="C3", label="marginal memoryless (converse)")
    ax.semilogy(R_bits, pe_ach, "s-", color="C1", label="optimal achievable prior")
    ax.semilogy(R_bits, pe_conv, "k-", lw=1, alpha=0.6, label="single-threshold (converse)")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel(r"$P_{exc}$")
    ax.set_title(f"G2  {TITLE}: prior gap"); ax.legend(fontsize=7.5)
    ax.grid(True, which="both", alpha=0.3)
    ax2.plot(R_bits, gain, "s-", color="C1")
    ax2.set_xlabel("rate $R$ (bits/sym)")
    ax2.set_ylabel("gain of optimal over best memoryless (%)")
    ax2.set_title("non-product prior gain"); ax2.grid(True, alpha=0.3)
    print(f"  G2: max gain {np.nanmax(gain):.2f}%")
    return save(fig, f"{PREFIX}_g2_prior_gap.png")


# ── G3 ───────────────────────────────────────────────────────────────────────
def g3_bounds_vs_exact():
    P_X, d_exc = _setup()
    osr = OneShotRD(P_X, d_exc)
    Y = d_exc.shape[1]
    curve = osr.compute_curve(np.ones(Y) / Y)
    R_bits = np.linspace(0.25, 1.5, 16)
    Ms = [np.exp(N * Rb * LN2) for Rb in R_bits]
    ex = [osr.theory(curve, M) for M in Ms]
    eb = [min(1.0, integrate_curve_rd_exp_bound(*curve, M)) for M in Ms]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(R_bits, ex, "o-", label="exact RC")
    ax.semilogy(R_bits, eb, "s--", label="exponential bound")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel(r"$P_{exc}$")
    ax.set_title(f"G3  {TITLE}: bound vs exact RC")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    return save(fig, f"{PREFIX}_g3_bounds_vs_exact.png")


# ── G4 ───────────────────────────────────────────────────────────────────────
def g4_fcurve_compare():
    P_X, d_exc = _setup()
    cover = (d_exc == 0).astype(float)
    osr = OneShotRD(P_X, d_exc)
    Rb = 0.8
    M = float(np.exp(N * Rb * LN2)); w0 = 1.0 / M
    P_conv, st_conv = osr.optimize_prior(M)
    pe_ach, P_ach = _Pexc_opt(P_X, cover, M)
    pe_conv = _pexc(P_X, cover, P_conv, M)

    ks, As = osr.compute_curve(P_conv)
    ka, Aa = osr.compute_curve(P_ach)
    w = np.linspace(0, min(1.0, 10 * w0), 400)

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ax.axvspan(0, w0, color="0.92", label=r"best-of-$M$ weight ($w\lesssim 1/M$)")
    ax.plot(w, np.interp(w, ks, As), color="C0", lw=2,
            label=f"converse-optimal prior  ($P_{{exc}}$={pe_conv:.2e})")
    ax.plot(w, np.interp(w, ka, Aa), color="C1", lw=2,
            label=f"achievability-optimal prior  ($P_{{exc}}$={pe_ach:.2e})")
    ax.axvline(w0, color="k", ls=":", lw=1.2)
    ax.set_xlabel("reproduction mass $w$"); ax.set_ylabel("excess spectrum $A(w)$")
    ax.set_title(f"G4  {TITLE}, $R={Rb}$: achievable vs converse prior")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)
    print(f"  G4: converse P_exc={pe_conv:.3e}, achievable P_exc={pe_ach:.3e}")
    return save(fig, f"{PREFIX}_g4_fcurve_compare.png")


# ── G5 ─ optimal achievable prior: full vs its i.i.d. product ─────────────────
def g5_full_vs_product():
    """How the exact excess probability changes when the achievability-optimal
    reproduction prior is replaced by its i.i.d. product version."""
    P_X, d_exc = _setup()
    cover = (d_exc == 0).astype(float)
    R_bits = np.linspace(0.3, 0.95, 9)
    af, ap = [], []
    for Rb in R_bits:
        M = float(np.exp(N * Rb * LN2))
        _, Q_ach = _Pexc_opt(P_X, cover, M)
        af.append(_pexc(P_X, cover, Q_ach, M))
        ap.append(_pexc(P_X, cover, _marginal_iid_lifted(Q_ach, N), M))
    af, ap = np.array(af), np.array(ap)
    cost = (ap - af) / af * 100.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax.semilogy(R_bits, af, "o-", color="C1", label="optimal achievable prior (full)")
    ax.semilogy(R_bits, ap, "^--", color="C2", label="its i.i.d. product (marginalized)")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel(r"achievable $P_{exc}$")
    ax.set_title(f"G5  {TITLE}: optimal prior vs its product")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    ax2.plot(R_bits, cost, "s-", color="C1")
    ax2.set_xlabel("rate $R$ (bits/sym)"); ax2.set_ylabel("marginalization cost (%)")
    ax2.set_title("cost of the i.i.d. product"); ax2.grid(True, alpha=0.3)
    print(f"  G5: max marginalization cost {np.nanmax(cost):.2f}%")
    return save(fig, f"{PREFIX}_g5_full_vs_product.png")


def main():
    for fn in (g1_mc_spread, g2_prior_gap, g3_bounds_vs_exact, g4_fcurve_compare,
               g5_full_vs_product):
        print(f"[{fn.__name__}]"); fn()


if __name__ == "__main__":
    main()
