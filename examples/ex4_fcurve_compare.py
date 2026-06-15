"""
F-curve comparison: converse-optimal prior vs achievability-optimal prior.

The converse and achievability bounds optimise the prior for *different*
objectives — the converse for a single threshold (the meta-converse LP), the
achievability for the whole kernel integral (the QP). Their optimal priors are
genuinely different, and so are their error spectra (F-curves). This figure
overlays the two F-curves at a fixed rate, with the kernel threshold marked.

(Requested figure: "the F curve of the converse prior optimization vs the F
curve of the optimal achievable curve.")
"""
import numpy as np
from _common import save, plt

from fbl import TypeBasedChannel
from fbl.channel_achievable_utils import z_channel
from fbl.prioropt import AchievabilityQP

N = 12
W = z_channel(0.1)
R_BITS = 0.25


def main():
    R = N * R_BITS * np.log(2.0)
    M = float(np.exp(R))
    w0 = 1.0 / M

    tbc = TypeBasedChannel(W, N)
    aqp = AchievabilityQP(W, N)

    # converse-optimal type prior and achievability-optimal type prior
    P_conv, pe_conv = tbc.optimize_prior(M)
    res = aqp.solve_rcu_plus(R)
    P_ach, pe_ach = res["Q_opt"], res["P_e_exact"]

    k_c, F_c = tbc.compute_curve(P_conv)
    k_a, F_a = tbc.compute_curve(P_ach)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.step(k_c, F_c, where="post", label=f"converse-optimal prior  ($P_e$={pe_conv:.3g})")
    ax.step(k_a, F_a, where="post", label=f"achievability-optimal prior  ($P_e$={pe_ach:.3g})")
    ax.axvline(w0, color="k", ls=":", lw=1, label=r"kernel threshold $w_0=1/M$")
    ax.set_xlabel(r"PEP value $w$")
    ax.set_ylabel(r"error spectrum $F(w)$")
    ax.set_xlim(0, min(1.0, 6 * w0))
    ax.set_title(f"Z(0.1), $n={N}$, $R={R_BITS}$ bits: converse vs achievability prior")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    print(f"  converse P_e={pe_conv:.4g}, achievability P_e={pe_ach:.4g}")
    return save(fig, "ex4_fcurve_compare.png")


if __name__ == "__main__":
    main()
