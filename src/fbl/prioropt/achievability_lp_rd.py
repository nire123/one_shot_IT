r"""
Exact RD achievability prior optimization: bracketing LP (ADDITIVE).

Minimizes the random-coding distortion D_rand(Q_Y, M) over the reproduction
type prior, on the verified distortion staircase of TypeBasedRateDistortion.
RD has no exact QP: the kernel antiderivative Phi(t)=1-(1-t)^M is a genuine
degree-M polynomial, so the realization is the bracketing LP (secant/tangent
PWL of the fixed Phi), the analog of Part-I prop:pwl-lp.

Per source-type block (tb.cond_x_y.iterate_cond()):
    sort conditional types by ASCENDING d_coeffs (P_X already baked in)
    delta_j = d_coeffs_sorted   (constants),  mass_j = Q[R_to_Q]*R_Q_ratio
    sigma_j = cumsum(mass)      (cvxpy, linear in Q),  sigma_m = 1, Phi(1)=1
Abel form (delta ascending so delta_j-delta_{j+1} <= 0):
    D = sum_block [ sum_{j<m} (delta_j-delta_{j+1}) Phi(sigma_j) + delta_m ],  minimized.
"""
import os
import sys


import numpy as np
import cvxpy as cp

from fbl.rd_achievable_type_based import TypeBasedRateDistortion
from fbl.F_curve import integrate_curve_rd_exact


def true_D_at_Q(tb, P_T_Y, M, num_refined_points=2000):
    """Exact best-of-M random-coding distortion at a reproduction type prior:
    D = M(M-1) int_0^1 A_X(w)(1-w)^{M-2} dw, on the type-based A-curve."""
    knots, A = tb.build_A_curve_type_based(P_T_Y)
    return integrate_curve_rd_exact(knots, A, M, num_refined_points=num_refined_points)


class AchievabilityLP_RD:
    def __init__(self, P_X_single, d_single, n, verbose=False):
        self.tb = TypeBasedRateDistortion(np.asarray(P_X_single, float),
                                          np.asarray(d_single, float),
                                          int(n), verbose=verbose)
        self.n = int(n)

    def _blocks(self):
        tb = self.tb
        out = []
        for (s, e) in tb.cond_x_y.iterate_cond():
            dc = tb.d_coeffs[s:e]
            order = np.argsort(dc)               # ASCENDING distortion (P_X baked in)
            out.append((dc[order],
                        tb.R_to_Q[s:e][order],
                        tb.R_Q_ratio[s:e][order]))
        return out

    def exact_D_rand(self, Q, M):
        """True best-of-M distortion at type prior Q (existing F-curve path)."""
        return float(true_D_at_Q(self.tb, np.asarray(Q, float), M))

    def solve_bracketing_lp(self, M, K=64):
        """
        Secant/tangent bracket of inf_Q D_rand(Q,M).
        Returns dict(D_lo, D_hi, gap, Q_lo, Q_hi); D_hi (secant) is the
        certified upper bound, D_lo (tangent) the lower bound.
        """
        Mf = float(M)
        Phi  = lambda t: 1.0 - np.maximum(1.0 - t, 0.0) ** Mf
        dPhi = lambda t: Mf * np.maximum(1.0 - t, 0.0) ** (Mf - 1.0)
        grid = np.linspace(0.0, 1.0, K + 1)
        Phi_g, dPhi_g = Phi(grid), dPhi(grid)
        blocks = self._blocks()

        def _solve(which):
            tb = self.tb
            Q = cp.Variable(tb.num_q, nonneg=True)
            cons = [cp.sum(Q) == 1.0]
            obj_terms = []
            const = 0.0
            for (delta, ridx, ratio) in blocks:
                m = len(delta)
                mass = cp.multiply(Q[ridx], ratio)
                sig, run = [], 0
                for j in range(m):
                    run = mass[j] if j == 0 else run + mass[j]
                    sig.append(run)
                const += float(delta[m - 1])              # j=m term: delta_m*Phi(1)
                delta_ext = np.concatenate([delta, [0.0]])
                for j in range(m - 1):                    # z terms j=1..m-1
                    coef = float(delta_ext[j] - delta_ext[j + 1])   # <= 0
                    if coef == 0.0:
                        continue
                    z = cp.Variable()
                    if which == "secant":
                        for k in range(1, len(grid)):
                            t0, t1 = grid[k - 1], grid[k]
                            slope = (Phi_g[k] - Phi_g[k - 1]) / (t1 - t0)
                            cons.append(z <= Phi_g[k - 1] + slope * (sig[j] - t0))
                    else:  # tangent
                        for k in range(len(grid)):
                            cons.append(z <= Phi_g[k] + dPhi_g[k] * (sig[j] - grid[k]))
                    obj_terms.append(coef * z)             # coef<=0, min pushes z up to Phi
            D = cp.sum(obj_terms) + const
            prob = cp.Problem(cp.Minimize(D), cons)
            prob.solve(solver=cp.CLARABEL)
            return float(D.value), np.asarray(Q.value, float)

        D_sec, Q_sec = _solve("secant")    # >= inf D  (upper)
        D_tan, Q_tan = _solve("tangent")   # <= inf D  (lower)
        return {"D_lo": D_tan, "D_hi": D_sec, "gap": D_sec - D_tan,
                "Q_lo": Q_tan, "Q_hi": Q_sec}
