"""
Joint source-channel coding (JSCC) — figures.

So far one figure (the prior gap); the full G1-G4 treatment is future work
(see results/jscc.md). For an i.i.d. source through a memoryless channel the
genuine non-product prior gain (QP vs the best memoryless prior at blocklength n)
is tiny; the larger gain vs the single-letter-optimal prior applied i.i.d. is a
within-memoryless artifact.

Run:  python examples/gen_jscc.py   ->   examples/figures/jscc_*.png
"""
import numpy as np
from _common import save, plt

from fbl.channel_achievable_utils import z_channel
from fbl.prioropt import AchievabilityJSCC

P_V = np.array([0.75, 0.25])
W = z_channel(0.1)
NS = [2, 3, 4, 5]
PREFIX = "jscc"


def g2_prior_gap():
    g_n1, g_best = [], []
    for n in NS:
        aj = AchievabilityJSCC(P_V, W, n)
        M = float(len(P_V) ** n)
        qp, _ = aj.solve_rcu_plus(M)
        mlo, _ = aj.memoryless_optimal(M)
        mlb, _ = aj.memoryless_baseline(M, n_starts=6, seed=1)
        g_n1.append(100 * (mlo - qp) / mlo)
        g_best.append(100 * (mlb - qp) / mlb)
        print(f"  n={n}: QP={qp:.6f} ml_opt(n1)={mlo:.6f} ml_best={mlb:.6f} "
              f"| vs_n1={g_n1[-1]:.2f}%  vs_best={g_best[-1]:.3f}%")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(NS, g_n1, "s-", color="C0", label="vs single-letter-optimal prior (i.i.d.)")
    ax.plot(NS, g_best, "o-", color="C3",
            label="vs best memoryless prior @ $n$  (true non-product gain)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("blocklength $n$"); ax.set_ylabel("QP gain (%)")
    ax.set_title("G2  JSCC (i.i.d. source): the non-product prior gain is tiny")
    ax.set_xticks(NS); ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
    return save(fig, f"{PREFIX}_g2_prior_gap.png")


def main():
    print("[g2_prior_gap]"); g2_prior_gap()


if __name__ == "__main__":
    main()
