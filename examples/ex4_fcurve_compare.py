"""
Error spectrum: converse-optimal prior vs achievability-optimal prior.

The converse and achievability bounds optimise the prior for *different*
objectives:

  * the meta-converse minimises the error at a **single threshold** (rate R);
  * the achievability bound integrates the error spectrum over the whole tail
    ``z >= R`` (weighted by e^{-Lz}), so it must keep the spectrum low *across*
    that region, not just at R.

We plot the **error spectrum** ``1 - F(w) = Pr[PEP > w] = Pr[-log PEP <= z]``
(with ``z = -log w``) on a log scale, for both optimal priors, at a fixed rate.
The converse-optimal prior dips lowest exactly at ``z = R`` and then blows up for
``z > R`` (it does not care about the tail); the achievability-optimal prior is a
hair higher at ``z = R`` but orders of magnitude lower across the shaded
integration region ``z >= R``.

(Requested figure: "the F curve of the converse prior optimization vs the F curve
of the optimal achievable curve.")
"""
import numpy as np
from _common import save, plt

from fbl import TypeBasedChannel
from fbl.channel_achievable_utils import z_channel
from fbl.prioropt import AchievabilityQP

N = 12
W = z_channel(0.1)
R_BITS = 0.25


def err_spectrum(knots, F, w):
    """1 - F(w) = Pr[PEP > w], clipped to stay positive on a log axis."""
    return np.clip(1.0 - np.interp(w, knots, F), 1e-12, None)


def main():
    R = N * R_BITS * np.log(2.0)          # total rate (nats); threshold z0 = R
    M = float(np.exp(R))

    tbc = TypeBasedChannel(W, N)
    aqp = AchievabilityQP(W, N)
    P_conv, pe_conv = tbc.optimize_prior(M)
    res = aqp.solve_rcu_plus(R)
    P_ach, pe_ach = res["Q_opt"], res["P_e_exact"]

    kc, Fc = tbc.compute_curve(P_conv)
    ka, Fa = tbc.compute_curve(P_ach)

    z = np.linspace(0.45 * R, 2.6 * R, 400)
    w = np.exp(-z)
    sc = err_spectrum(kc, Fc, w)
    sa = err_spectrum(ka, Fa, w)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axvspan(R, z.max(), color="0.92",
               label=r"achievability integrates here ($z\geq R$)")
    ax.semilogy(z, sc, color="C0", lw=2,
                label=f"converse-optimal prior  (bound {pe_conv:.1e})")
    ax.semilogy(z, sa, color="C1", lw=2,
                label=f"achievability-optimal prior  (bound {pe_ach:.1e})")
    ax.axvline(R, color="k", ls=":", lw=1.2)
    ax.annotate("threshold z = R", xy=(R, sc.min() * 3),
                xytext=(R * 1.02, sc.min() * 30), fontsize=8.5)
    ax.set_xlabel("z = -log(PEP)   (nats)")
    ax.set_ylabel("error spectrum   Pr[PEP > w] = Pr[-log PEP <= z]")
    ax.set_title(f"Z(0.1), $n={N}$, $R={R_BITS}$ bits/use: converse vs achievability prior")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    print(f"  converse prior: lowest at z=R ({pe_conv:.2e}); blows up for z>R")
    print(f"  achievability prior: low across z>=R (bound {pe_ach:.2e})")
    return save(fig, "ex4_fcurve_compare.png")


if __name__ == "__main__":
    main()
