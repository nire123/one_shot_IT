"""
Rate-distortion (EXCESS distortion) — the four thesis figures.

Excess distortion is the probability that the block distortion exceeds a
threshold, ``P_exc = Pr[d(X, X_hat) > T]``.  For a random size-M codebook this is
exactly the best-of-M of the **block-distortion indicator** ``d_exc = 1{d > T}``:

    P_exc(Q) = E_X[ min_{y in C} 1{d(X,y) > T} ] = sum_x P_X(x) (1 - q(x))^M,
    q(x) = Q{ y : d(x,y) <= T }   (the per-source coverage probability).

So excess distortion is the ordinary lossy-source-coding machinery applied to the
0/1 indicator distortion -- no new derivation.  We work in the lifted X^n space
(exact, small n), with binary memoryless source + block-Hamming distortion.

  G1  Monte-Carlo spread of realised P_exc vs the analytic expectation
  G2  prior gap: achievability-optimal reproduction prior vs best memoryless
  G3  exact P_exc vs the exponential surrogate
  G4  coverage spectrum of the converse-optimal vs achievability-optimal prior

Run:  python examples/gen_rd_excess.py   ->   examples/figures/rd_exc_*.png
"""
import numpy as np
import cvxpy as cp
from _common import save, plt

from fbl import OneShotRD
from fbl.achievable_utils import binary_memoryless_source, hamming_distortion
from fbl.F_curve import integrate_curve_rd_exp_bound

P = 0.25                      # source bias
N = 6                         # blocklength (lifted, exact)
T = 1                         # excess threshold on block Hamming: P_exc = Pr[Hamming > 1]
PREFIX = "rd_exc"
TITLE = f"RD excess distortion (BMS p={P}, $n={N}$, $T={T}$)"


def _setup():
    P_X = binary_memoryless_source(P, N)
    H = hamming_distortion(N)
    d_exc = (H > T).astype(float)            # block-distortion indicator
    return P_X, d_exc


def _Pexc_memoryless(P_X, cover, q1, M):
    """Exact P_exc for the i.i.d. reproduction prior with single-letter bias q1."""
    n = int(round(np.log2(len(P_X))))
    Qy = np.array([q1[1] ** bin(y).count("1") * q1[0] ** (n - bin(y).count("1"))
                   for y in range(len(P_X))])
    Qy /= Qy.sum()
    q = cover @ Qy
    return float(np.sum(P_X * (1 - q) ** M))


def _Pexc_opt(P_X, cover, M):
    """Achievability-optimal reproduction prior over Y^n (convex)."""
    Y = cover.shape[1]
    Q = cp.Variable(Y, nonneg=True)
    q = cover @ Q
    obj = cp.sum(cp.multiply(P_X, cp.power(1 - q, M)))
    cp.Problem(cp.Minimize(obj), [cp.sum(Q) == 1]).solve(solver=cp.CLARABEL)
    return float(obj.value), np.asarray(Q.value, float)


# ── G1 ───────────────────────────────────────────────────────────────────────
def g1_mc_spread():
    P_X, d_exc = _setup()
    osr = OneShotRD(P_X, d_exc)
    Y = d_exc.shape[1]
    Q = np.ones(Y) / Y
    curve = osr.compute_curve(Q)
    R_bits = np.linspace(0.25, 0.85, 12)      # cap: MC cannot estimate rarer events
    Ms = [max(2, int(round(np.exp(N * Rb * np.log(2))))) for Rb in R_bits]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _ in range(60):
        pe = [max(osr.evaluate(osr.draw_random_code(Q, M, rng)), 1e-6) for M in Ms]
        ax.semilogy(R_bits, pe, color="C0", alpha=0.10, lw=0.8)
    ax.plot([], [], color="C0", alpha=0.4, label="60 random codebooks")
    exp = [osr.theory(curve, M) for M in Ms]
    ax.semilogy(R_bits, exp, "k-", lw=2.2, label="expectation (exact)")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel(r"excess prob. $P_{exc}$")
    ax.set_title(f"G1  {TITLE}: random-code spread")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    return save(fig, f"{PREFIX}_g1_mc_spread.png")


# ── G2 ───────────────────────────────────────────────────────────────────────
def g2_prior_gap():
    P_X, d_exc = _setup()
    cover = (d_exc == 0).astype(float)
    osr = OneShotRD(P_X, d_exc)
    qgrid = np.linspace(0.04, 0.96, 21)
    R_bits = np.linspace(0.3, 0.95, 9)        # reliable range (converse LP above solver floor)

    pe_opt, pe_ml, pe_conv = [], [], []
    for Rb in R_bits:
        M = float(np.exp(N * Rb * np.log(2)))
        pe_opt.append(_Pexc_opt(P_X, cover, M)[0])
        pe_ml.append(min(_Pexc_memoryless(P_X, cover, np.array([1 - q, q]), M) for q in qgrid))
        pe_conv.append(osr.optimize_prior(M)[1])
    pe_opt, pe_ml, pe_conv = map(np.array, (pe_opt, pe_ml, pe_conv))
    gain = (pe_ml - pe_opt) / pe_ml * 100.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax.semilogy(R_bits, pe_ml, "o-", label="best memoryless prior")
    ax.semilogy(R_bits, pe_opt, "s-", label="achievability-optimal prior")
    ax.semilogy(R_bits, pe_conv, "k--", label="single-threshold LP")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel(r"$P_{exc}$")
    ax.set_title(f"G2  {TITLE}: prior gap"); ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax2.plot(R_bits, gain, "s-", color="C1")
    ax2.set_xlabel("rate $R$ (bits/sym)"); ax2.set_ylabel("gain over memoryless (%)")
    ax2.set_title("prior-family gap"); ax2.grid(True, alpha=0.3)
    print(f"  G2 max gain over memoryless: {np.nanmax(gain):.2f}%")
    return save(fig, f"{PREFIX}_g2_prior_gap.png")


# ── G3 ───────────────────────────────────────────────────────────────────────
def g3_bounds_vs_exact():
    P_X, d_exc = _setup()
    osr = OneShotRD(P_X, d_exc)
    Y = d_exc.shape[1]
    curve = osr.compute_curve(np.ones(Y) / Y)
    R_bits = np.linspace(0.25, 1.5, 16)
    Ms = [np.exp(N * Rb * np.log(2)) for Rb in R_bits]
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
    M = float(np.exp(N * Rb * np.log(2)))
    w0 = 1.0 / M

    P_conv, st_conv = osr.optimize_prior(M)            # single-threshold prior + value
    pe_ach, P_ach = _Pexc_opt(P_X, cover, M)           # achievability-optimal prior
    pe_conv = float(np.sum(P_X * (1 - cover @ P_conv) ** M))

    ks, As = osr.compute_curve(P_conv)
    ka, Aa = osr.compute_curve(P_ach)
    w = np.linspace(0, min(1.0, 10 * w0), 400)

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ax.axvspan(0, w0, color="0.92", label=r"best-of-$M$ weight ($w\lesssim 1/M$)")
    ax.plot(w, np.interp(w, ks, As), color="C0", lw=2,
            label=f"single-threshold prior\n   LP={st_conv:.2e}  |  $P_{{exc}}$={pe_conv:.2e}")
    ax.plot(w, np.interp(w, ka, Aa), color="C1", lw=2,
            label=f"achievability-optimal prior\n   $P_{{exc}}$={pe_ach:.2e}  (min)")
    ax.axvline(w0, color="k", ls=":", lw=1.2)
    ax.set_xlabel("reproduction mass $w$"); ax.set_ylabel("excess spectrum $A(w)$")
    ax.set_title(f"G4  {TITLE}, $R={Rb}$: converse vs achievability prior")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)
    print(f"  G4 converse P_exc={pe_conv:.3e}, achievability P_exc={pe_ach:.3e}")
    return save(fig, f"{PREFIX}_g4_fcurve_compare.png")


def main():
    for fn in (g1_mc_spread, g2_prior_gap, g3_bounds_vs_exact, g4_fcurve_compare):
        print(f"[{fn.__name__}]"); fn()


if __name__ == "__main__":
    main()
