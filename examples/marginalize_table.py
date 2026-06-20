"""Marginalization table (companion to the G5 figures).

For each use case, report the **achievable bound** obtained by plugging in four
priors: the achievability-optimal prior and the converse-optimal prior, each in
its full form and as its i.i.d. per-symbol marginal. This answers: how does the
achievable bound change when we marginalize the converse prior, or the
achievable prior? Parameters match the corresponding G5 figure.

Run:  python examples/marginalize_table.py
"""
import numpy as np

import gen_channel as gc        # imports _common -> sets sys.path for fbl
import gen_rd_average as ga
import gen_rd_excess as ge

from fbl import TypeBasedChannel, TypeBasedRD, OneShotRD
from fbl.type_based_utils import memoryless_to_type_prior
from fbl.prioropt import AchievabilityQP, AchievabilityLP_RD, rcu_plus_from_F_curve
from fbl.prioropt.typebased_block_lp import marginal_input

LN2 = np.log(2.0)


def channel_row():
    W, n, Rb = gc.W, 12, 0.25                      # matches gen_channel.g5
    R = n * Rb * LN2; M = np.exp(R); w0 = 1.0 / M
    tbc = TypeBasedChannel(W, n); aqp = AchievabilityQP(W, n)
    P_conv = tbc.optimize_prior(M)[0]
    P_ach = aqp.solve_rcu_plus(R)["Q_opt"]
    marg = lambda P: memoryless_to_type_prior(marginal_input(P, n, W.shape[0]), n)
    bnd = lambda P: rcu_plus_from_F_curve(*tbc.compute_curve(P), w0)
    return ("channel Z(0.1)", "P_e", bnd(P_ach), bnd(marg(P_ach)),
            bnd(P_conv), bnd(marg(P_conv)))


def rd_avg_row():
    P, D, n, Rb = ga.P, ga.D_SINGLE, 8, 0.4        # matches gen_rd_average.g5
    M = np.exp(n * Rb * LN2)
    alr = AchievabilityLP_RD([1 - P, P], D, n)
    tbr = TypeBasedRD([1 - P, P], D, n)
    P_conv = tbr.optimize_prior(M)[0]
    P_ach = alr.solve_bracketing_lp(M, K=32)["Q_hi"]
    marg = lambda Pr: memoryless_to_type_prior(marginal_input(Pr, n, D.shape[1]), n)
    bnd = lambda Pr: alr.exact_D_rand(Pr, M) / n
    return ("RD average", "D/sym", bnd(P_ach), bnd(marg(P_ach)),
            bnd(P_conv), bnd(marg(P_conv)))


def rd_exc_row():
    P_X, d_exc = ge._setup()                        # matches gen_rd_excess.g5
    cover = (d_exc == 0).astype(float)
    osr = OneShotRD(P_X, d_exc)
    N, Rb = ge.N, 0.8
    M = np.exp(N * Rb * LN2)
    P_conv = osr.optimize_prior(M)[0]
    P_ach = ge._Pexc_opt(P_X, cover, M)[1]
    marg = lambda Q: ge._marginal_iid_lifted(Q, N)
    bnd = lambda Q: float(np.sum(P_X * (1 - cover @ Q) ** M))
    return ("RD excess", "P_exc", bnd(P_ach), bnd(marg(P_ach)),
            bnd(P_conv), bnd(marg(P_conv)))


def main():
    rows = [channel_row(), rd_avg_row(), rd_exc_row()]
    g = lambda x: f"{x:.3g}"
    print("| use case | metric | achiev-opt (full) | achiev-opt (marginal) "
          "| converse (full) | converse (marginal) |")
    print("|---|---|---|---|---|---|")
    for name, metric, af, am, cf, cm in rows:
        print(f"| {name} | {metric} | {g(af)} | {g(am)} | {g(cf)} | {g(cm)} |")


if __name__ == "__main__":
    main()
