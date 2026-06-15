"""
G2 — the prior gap (channel coding).

The exact achievability-optimal prior (QP over all type priors) vs the best
memoryless prior vs the meta-converse, as a function of rate, at fixed n.
This is the figure the library is *for*: it measures how much error probability
the memoryless prior family costs you, against the true achievability optimum
(not a heuristic).
"""
import numpy as np
from _common import save, plt

from fbl import TypeBasedChannel
from fbl.type_based_utils import memoryless_to_type_prior
from fbl.channel_achievable_utils import z_channel
from fbl.prioropt import AchievabilityQP
from fbl.prioropt.typebased_block_lp import rcu_plus_from_F_curve

N = 12
W = z_channel(0.1)
Q_GRID = np.linspace(0.04, 0.96, 25)
R_BITS = np.linspace(0.06, 0.9, 16)


def main():
    aqp = AchievabilityQP(W, N)
    tbc = TypeBasedChannel(W, N)
    ml_curves = [tbc.compute_curve(memoryless_to_type_prior(np.array([1 - q, q]), N))
                 for q in Q_GRID]

    pe_qp, pe_ml, pe_conv = [], [], []
    for Rb in R_BITS:
        R = N * Rb * np.log(2.0)
        M = float(np.exp(R))
        pe_qp.append(aqp.solve_rcu_plus(R)["P_e_exact"])
        pe_ml.append(min(rcu_plus_from_F_curve(*c, 1.0 / M) for c in ml_curves))
        pe_conv.append(tbc.optimize_prior(M)[1])

    pe_qp, pe_ml, pe_conv = map(np.array, (pe_qp, pe_ml, pe_conv))
    gain = (pe_ml - pe_qp) / pe_ml * 100.0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax.semilogy(R_BITS, pe_ml, "o-", label="best memoryless prior")
    ax.semilogy(R_BITS, pe_qp, "s-", label="optimal achievability prior (QP)")
    ax.semilogy(R_BITS, pe_conv, "k--", label="meta-converse")
    ax.set_xlabel("rate $R$ (bits/use)")
    ax.set_ylabel(r"$P_e$")
    ax.set_title(f"Z(0.1) channel, $n={N}$: achievability prior gap")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax2.plot(R_BITS, gain, "s-", color="C1")
    ax2.set_xlabel("rate $R$ (bits/use)")
    ax2.set_ylabel("QP gain over memoryless (%)")
    ax2.set_title("prior-family gap (low-rate corner)")
    ax2.grid(True, alpha=0.3)

    print(f"  max QP gain over memoryless: {gain.max():.2f}% at R={R_BITS[gain.argmax()]:.2f} bits")
    print(f"  max(QP - converse) sanity (>=0): {np.max(pe_qp - pe_conv):.2e}")
    return save(fig, "ex2_prior_gap.png")


if __name__ == "__main__":
    main()
