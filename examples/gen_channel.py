"""
Channel coding — the four thesis figures.

  G1  Monte-Carlo spread of realised error vs the RCU expectation
  G2  prior gap: achievability-optimal prior (QP) vs best memoryless
  G3  exact random-coding error vs the union-bound surrogate
  G4  error spectrum of the converse-optimal vs achievability-optimal prior

Run:  python examples/gen_channel.py   ->   examples/figures/channel_*.png
"""
import numpy as np
from _common import save, plt

from fbl import OneShotChannel, TypeBasedChannel
from fbl.type_based_utils import memoryless_to_type_prior
from fbl.channel_achievable_utils import z_channel, kronecker_power
from fbl.F_curve import (integrate_curve_channel_coding_exact,
                         integrate_curve_channel_coding_union_bound)
from fbl.prioropt import AchievabilityQP, rcu_plus_from_F_curve

W = z_channel(0.1)
PREFIX = "channel"
TITLE = "channel Z(0.1)"


# ── G1 ─ Monte-Carlo spread vs RCU expectation ───────────────────────────────
def g1_mc_spread():
    n = 6
    osc = OneShotChannel(kronecker_power(W, n))
    q1 = np.array([0.6, 0.4])
    Q = np.array([q1[1] ** bin(x).count("1") * q1[0] ** (n - bin(x).count("1"))
                  for x in range(2 ** n)])
    Q /= Q.sum()
    curve = osc.compute_curve(Q)
    R_bits = np.linspace(0.12, 0.78, 12)
    Ms = [max(2, int(round(np.exp(n * Rb * np.log(2))))) for Rb in R_bits]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _ in range(60):
        pe = [osc.evaluate(osc.draw_random_code(Q, M, rng)) for M in Ms]
        ax.plot(R_bits, pe, color="C0", alpha=0.10, lw=0.8)
    ax.plot([], [], color="C0", alpha=0.4, label="60 random codebooks")
    ax.plot(R_bits, [osc.theory(curve, M) for M in Ms], "k-", lw=2.2,
            label="RCU expectation (exact RC)")
    ax.set_xlabel("rate $R$ (bits/use)"); ax.set_ylabel(r"$P_e$")
    ax.set_title(f"G1  {TITLE}, $n={n}$: random-code spread")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return save(fig, f"{PREFIX}_g1_mc_spread.png")


# ── G2 ─ prior gap ───────────────────────────────────────────────────────────
def g2_prior_gap():
    n = 12
    aqp = AchievabilityQP(W, n)
    tbc = TypeBasedChannel(W, n)
    ml_curves = [tbc.compute_curve(memoryless_to_type_prior(np.array([1 - q, q]), n))
                 for q in np.linspace(0.04, 0.96, 25)]
    R_bits = np.linspace(0.06, 0.9, 12)

    pe_qp, pe_ml, pe_conv = [], [], []
    for Rb in R_bits:
        R = n * Rb * np.log(2.0); M = float(np.exp(R))
        pe_qp.append(aqp.solve_rcu_plus(R)["P_e_exact"])
        pe_ml.append(min(rcu_plus_from_F_curve(*c, 1.0 / M) for c in ml_curves))
        pe_conv.append(tbc.optimize_prior(M)[1])
    pe_qp, pe_ml, pe_conv = map(np.array, (pe_qp, pe_ml, pe_conv))
    gain = (pe_ml - pe_qp) / pe_ml * 100.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax.semilogy(R_bits, pe_ml, "o-", label="best memoryless prior")
    ax.semilogy(R_bits, pe_qp, "s-", label="achievability-optimal prior (QP)")
    ax.semilogy(R_bits, pe_conv, "k--", label="meta-converse")
    ax.set_xlabel("rate $R$ (bits/use)"); ax.set_ylabel(r"$P_e$")
    ax.set_title(f"G2  {TITLE}, $n={n}$: prior gap"); ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax2.plot(R_bits, gain, "s-", color="C1")
    ax2.set_xlabel("rate $R$ (bits/use)"); ax2.set_ylabel("QP gain over memoryless (%)")
    ax2.set_title("prior-family gap (low-rate corner)"); ax2.grid(True, alpha=0.3)
    print(f"  G2 max QP gain: {gain.max():.2f}% at R={R_bits[gain.argmax()]:.2f} bits")
    return save(fig, f"{PREFIX}_g2_prior_gap.png")


# ── G3 ─ exact RC vs union bound ─────────────────────────────────────────────
def g3_bounds_vs_exact():
    n = 10
    tbc = TypeBasedChannel(W, n)
    cc = tbc.compute_curve(memoryless_to_type_prior(np.array([0.55, 0.45]), n))
    R_bits = np.linspace(0.1, 0.85, 16)
    Ms = [np.exp(n * Rb * np.log(2)) for Rb in R_bits]
    ex = [integrate_curve_channel_coding_exact(*cc, M) for M in Ms]
    ub = [integrate_curve_channel_coding_union_bound(*cc, M) for M in Ms]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogy(R_bits, ex, "o-", label="exact RC")
    ax.semilogy(R_bits, ub, "s--", label="union bound")
    ax.set_xlabel("rate $R$ (bits/use)"); ax.set_ylabel(r"$P_e$")
    ax.set_title(f"G3  {TITLE}, $n={n}$: bound vs exact RC")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
    return save(fig, f"{PREFIX}_g3_bounds_vs_exact.png")


# ── G4 ─ error spectrum: converse- vs achievability-optimal prior ────────────
def g4_fcurve_compare():
    n = 12
    Rb = 0.25
    R = n * Rb * np.log(2.0); M = float(np.exp(R)); w0 = 1.0 / M
    tbc = TypeBasedChannel(W, n)
    aqp = AchievabilityQP(W, n)
    P_conv, _ = tbc.optimize_prior(M)
    res = aqp.solve_rcu_plus(R); P_ach = res["Q_opt"]

    kc, Fc = tbc.compute_curve(P_conv)
    ka, Fa = tbc.compute_curve(P_ach)
    atR_c = 1.0 - np.interp(w0, kc, Fc); atR_a = 1.0 - np.interp(w0, ka, Fa)
    bnd_c = rcu_plus_from_F_curve(kc, Fc, w0); bnd_a = rcu_plus_from_F_curve(ka, Fa, w0)

    def espec(k, F, w):
        return np.clip(1.0 - np.interp(w, k, F), 1e-12, None)
    z = np.linspace(0.45 * R, 2.6 * R, 400); w = np.exp(-z)

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ax.axvspan(R, z.max(), color="0.92", label=r"achievability integrates here ($z\geq R$)")
    ax.semilogy(z, espec(kc, Fc, w), color="C0", lw=2,
                label=f"converse-optimal prior\n   @R={atR_c:.1e} (min)  |  bound={bnd_c:.1e}")
    ax.semilogy(z, espec(ka, Fa, w), color="C1", lw=2,
                label=f"achievability-optimal prior\n   @R={atR_a:.1e}  |  bound={bnd_a:.1e} (min)")
    ax.axvline(R, color="k", ls=":", lw=1.2)
    ax.set_xlabel("z = -log(PEP)   (nats)")
    ax.set_ylabel("error spectrum   Pr[PEP > w] = Pr[-log PEP <= z]")
    ax.set_title(f"G4  {TITLE}, $n={n}$, $R={Rb}$ bits: converse vs achievability prior")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, which="both", alpha=0.3)
    print(f"  G4 converse bound={bnd_c:.2e}, achievability bound={bnd_a:.2e}")
    return save(fig, f"{PREFIX}_g4_fcurve_compare.png")


def main():
    for fn in (g1_mc_spread, g2_prior_gap, g3_bounds_vs_exact, g4_fcurve_compare):
        print(f"[{fn.__name__}]"); fn()


if __name__ == "__main__":
    main()
