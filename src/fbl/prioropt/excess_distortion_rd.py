r"""
Excess-distortion prior optimization (ADDITIVE).

Key fact: the excess-distortion probability of a random size-M codebook,
    P_exc(Q_Y) = E_X[ min_{y in C} 1{d(X,y) > d_th} ] = sum_x P_X(x) (1 - q(x))^M,
    q(x) = Q_Y{y : d(x,y) <= d_th},
is exactly the BEST-OF-M of the INDICATOR distortion d_e = 1{d > d_th}.  So the
excess prior optimization is the average-distortion machinery
(`AchievabilityLP_RD`) applied to d_e -- no new derivation.

This module provides:
  - ExcessRD: a thin wrapper (build d_e, reuse AchievabilityLP_RD)
  - exact P_exc at a prior, the bracketing-LP optimum, and helpers to compare
    against the best memoryless prior and the chord-rule block-LP prior.
"""
import os
import sys


import numpy as np

from fbl.prioropt.achievability_lp_rd import AchievabilityLP_RD
from fbl.rd_achievable_type_based import TypeBasedRateDistortion
from fbl.type_based_utils import memoryless_to_type_prior
from fbl.F_curve import integrate_curve_rd_exact
from fbl.prioropt.typebased_block_lp_rd import TypeBasedBlockLPRD, grid_around_zstar, true_D_at_Q


class ExcessRD:
    def __init__(self, P_X_single, d_single, d_th, n):
        self.P_X = np.asarray(P_X_single, float)
        self.d = np.asarray(d_single, float)
        self.d_th = float(d_th)
        self.n = int(n)
        self.d_e = (self.d > self.d_th).astype(float)        # indicator distortion
        self.alp = AchievabilityLP_RD(self.P_X, self.d_e, n)
        self.tb = self.alp.tb

    # exact excess-distortion probability at a reproduction TYPE prior
    def exact_P_exc(self, Q_type, M):
        return self.alp.exact_D_rand(Q_type, M)

    # bracketing-LP optimum over all reproduction type priors
    def solve_bracketing_lp(self, M, K=96):
        return self.alp.solve_bracketing_lp(M, K)


def memoryless_to_type(Q_single, n):
    return memoryless_to_type_prior(np.asarray(Q_single, float), n)
