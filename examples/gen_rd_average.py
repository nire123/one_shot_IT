"""
Rate-distortion (average distortion) — the four thesis figures.

Mirrors the channel-coding suite for the lossy-source-coding setting (binary
memoryless source + Hamming distortion), average-distortion flavour:

  G1  Monte-Carlo spread of realised distortion vs the RCB expectation
  G2  prior gap: achievability-optimal reproduction prior vs best memoryless
  G3  exact random-coding distortion vs the exponential surrogate
  G4  distortion spectrum of the converse-optimal vs achievability-optimal prior

Run:  python examples/gen_rd_average.py   ->   examples/figures/rd_avg_*.png
"""
import numpy as np
from _common import save, plt

from fbl import OneShotRD, TypeBasedRD
from fbl.achievable_utils import setup_bms_hamming
from fbl.type_based_utils import memoryless_to_type_prior
from fbl.F_curve import integrate_curve_rd_exact, integrate_curve_rd_exp_bound
from fbl.prioropt import AchievabilityLP_RD

P = 0.25                                  # source bias  P_X = [1-p, p] (asymmetric)
D_SINGLE = np.array([[0.0, 1.0], [1.0, 0.0]])   # single-letter Hamming
PREFIX = "rd_avg"
TITLE = "RD avg. distortion (BMS p=0.25 + Hamming)"


# ── G1 ─ Monte-Carlo spread vs RCB expectation ───────────────────────────────
def g1_mc_spread():
    n = 7
    P_X, d, _ = setup_bms_hamming(P, n)
    osr = OneShotRD(P_X, d)
    Y = d.shape[1]
    Q = np.ones(Y) / Y                    # uniform reproduction prior
    curve = osr.compute_curve(Q)
    R_bits = np.linspace(0.08, 0.85, 12)
    Ms = [max(2, int(round(np.exp(n * Rb * np.log(2))))) for Rb in R_bits]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _ in range(60):
        dd = [osr.evaluate(osr.draw_random_code(Q, M, rng)) / n for M in Ms]
        ax.plot(R_bits, dd, color="C0", alpha=0.10, lw=0.8)
    ax.plot([], [], color="C0", alpha=0.4, label="60 random codebooks")
    exp = [osr.theory(curve, M) / n for M in Ms]
    ax.plot(R_bits, exp, "k-", lw=2.2, label="RCB expectation (exact RC)")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel("distortion per symbol")
    ax.set_title(f"G1  {TITLE}, $n={n}$: random-code spread")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return save(fig, f"{PREFIX}_g1_mc_spread.png")


# ── G2 ─ prior gap ───────────────────────────────────────────────────────────
def g2_prior_gap():
    n = 8
    alr = AchievabilityLP_RD(np.array([1 - P, P]), D_SINGLE, n)
    tbr = TypeBasedRD(np.array([1 - P, P]), D_SINGLE, n)
    qgrid = np.linspace(0.04, 0.96, 21)
    ml_curves = [tbr.compute_curve(memoryless_to_type_prior(np.array([1 - q, q]), n))
                 for q in qgrid]
    R_bits = np.linspace(0.18, 0.7, 8)            # 2 <= M <= ~50 (bracket accurate)

    d_opt, d_ml, d_st = [], [], []
    for Rb in R_bits:
        M = float(np.exp(n * Rb * np.log(2)))
        br = alr.solve_bracketing_lp(M, K=24)
        d_opt.append(min(alr.exact_D_rand(br["Q_hi"], M),
                         alr.exact_D_rand(br["Q_lo"], M)) / n)
        d_ml.append(min(tbr.theory(c, M) for c in ml_curves) / n)
        d_st.append(tbr.optimize_prior(M)[1] / n)
    d_opt, d_ml, d_st = map(np.array, (d_opt, d_ml, d_st))
    gain = (d_ml - d_opt) / d_ml * 100.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax.plot(R_bits, d_ml, "o-", label="best memoryless prior")
    ax.plot(R_bits, d_opt, "s-", label="achievability-optimal prior (LP)")
    ax.plot(R_bits, d_st, "k--", label="single-threshold LP")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel("distortion per symbol")
    ax.set_title(f"G2  {TITLE}, $n={n}$: prior gap")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax2.plot(R_bits, gain, "s-", color="C1")
    ax2.set_xlabel("rate $R$ (bits/sym)"); ax2.set_ylabel("LP gain over memoryless (%)")
    ax2.set_title("prior-family gap"); ax2.grid(True, alpha=0.3)
    print(f"  G2 max LP gain over memoryless: {gain.max():.2f}%")
    return save(fig, f"{PREFIX}_g2_prior_gap.png")


# ── G3 ─ exact RC vs exponential surrogate ───────────────────────────────────
def g3_bounds_vs_exact():
    n = 10
    tbr = TypeBasedRD(np.array([1 - P, P]), D_SINGLE, n)
    curve = tbr.compute_curve(memoryless_to_type_prior(np.array([0.5, 0.5]), n))
    R_bits = np.linspace(0.1, 0.85, 16)
    Ms = [np.exp(n * Rb * np.log(2)) for Rb in R_bits]
    ex = [integrate_curve_rd_exact(*curve, M) / n for M in Ms]
    eb = [integrate_curve_rd_exp_bound(*curve, M) / n for M in Ms]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(R_bits, ex, "o-", label="exact RC")
    ax.plot(R_bits, eb, "s--", label="exponential bound")
    ax.set_xlabel("rate $R$ (bits/sym)"); ax.set_ylabel("distortion per symbol")
    ax.set_title(f"G3  {TITLE}, $n={n}$: bound vs exact RC")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return save(fig, f"{PREFIX}_g3_bounds_vs_exact.png")


# ── G4 ─ distortion spectrum: converse- vs achievability-optimal prior ────────
def g4_fcurve_compare():
    n = 8
    Rb = 0.4
    M = float(np.exp(n * Rb * np.log(2)))
    w0 = 1.0 / M
    alr = AchievabilityLP_RD(np.array([1 - P, P]), D_SINGLE, n)
    tbr = TypeBasedRD(np.array([1 - P, P]), D_SINGLE, n)

    P_st, _ = tbr.optimize_prior(M)                    # single-threshold (converse) prior
    br = alr.solve_bracketing_lp(M, K=32)
    P_ach = br["Q_hi"]
    # actual best-of-M distortion of EACH prior (the achievability metric)
    d_st = alr.exact_D_rand(P_st, M)
    d_ach = alr.exact_D_rand(P_ach, M)

    ks, As = tbr.compute_curve(P_st)
    ka, Aa = tbr.compute_curve(P_ach)
    # per-symbol distortion-at-threshold A(w0) (the single-threshold metric)
    atR_st = np.interp(w0, ks, As) / n
    atR_ach = np.interp(w0, ka, Aa) / n

    w = np.linspace(0, min(1.0, 8 * w0), 400)
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ax.axvspan(0, w0, color="0.92", label=r"best-of-$M$ weight concentrates ($w\lesssim 1/M$)")
    ax.plot(w, np.interp(w, ks, As) / n, color="C0", lw=2,
            label=f"single-threshold prior\n   A($w_0$)={atR_st:.3f}   |   D={d_st/n:.3f}")
    ax.plot(w, np.interp(w, ka, Aa) / n, color="C1", lw=2,
            label=f"achievability-optimal prior\n   A($w_0$)={atR_ach:.3f}   |   D={d_ach/n:.3f}  (min)")
    ax.axvline(w0, color="k", ls=":", lw=1.2)
    ax.annotate("$w_0=1/M$", xy=(w0, 0.02), fontsize=8.5)
    ax.set_xlabel("reproduction mass $w$"); ax.set_ylabel("distortion spectrum $A(w)$ per symbol")
    ax.set_title(f"G4  {TITLE}, $n={n}$, $R={Rb}$: converse vs achievability prior")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
    print(f"  G4 single-threshold D={d_st/n:.4f}, achievability D={d_ach/n:.4f}")
    return save(fig, f"{PREFIX}_g4_fcurve_compare.png")


def main():
    for fn in (g1_mc_spread, g2_prior_gap, g3_bounds_vs_exact, g4_fcurve_compare):
        print(f"[{fn.__name__}]"); fn()


if __name__ == "__main__":
    main()
