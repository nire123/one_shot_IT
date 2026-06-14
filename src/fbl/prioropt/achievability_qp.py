r"""
Exact achievability prior optimization: convex QP (RCU+) and bracketing LP.

ADDITIVE module. Imports the existing TypeBasedChannel machinery unchanged and
builds, on the same type data, the exact convex quadratic program of
`thm:qp` (Appendix F of the PEP-framework paper) and the general-kernel
bracketing LP of `prop:pwl-lp`. This replaces the *approximate* kink-adaptive
chord-rule `TypeBasedBlockLP` with the *exact* program.

Structural map (paper single-letter  ->  repo type-based), per output-type block
(s,e) from `tb.cond_y_x.iterate_cond()`:
    inputs sorted by W(y|x) desc   ->  argsort(alpha_coeffs[s:e])[::-1]   (fixed)
    channel value nu^y_j            ->  alpha_coeffs[s:e][order]           (const)
    mass q_i = Q_X(x)               ->  Q[R_to_Q[s:e][order]] * R_Q_ratio  (linear in Q)
    knot sigma^y_j                  ->  cumsum(mass)                       (cvxpy expr)

Rate convention: R is the TOTAL rate (M = e^R competitors), w0 = e^{-R} = 1/M,
matching `rcu_plus_from_F_curve(w_max=1/M)` and `plot_lp_vs_memoryless_channel`.
"""

import os
import sys


import numpy as np
import cvxpy as cp

from fbl.channel_achievable_type_based import TypeBasedChannel


def _phi_exact(t, M):
    """Exact-RC antiderivative Phi(t) = (1-(1-t)^M)/M (M = e^R + 1 here)."""
    base = np.maximum(1.0 - t, 0.0)
    return (1.0 - base ** M) / M


def _dphi_exact(t, M):
    """Phi'(t) = (1-t)^{M-1}."""
    return np.maximum(1.0 - t, 0.0) ** (M - 1.0)


class AchievabilityQP:
    """
    Exact achievability prior optimization on a type-based channel.

    Parameters
    ----------
    W_single : (k_x, k_y) ndarray   single-letter channel.
    n : int                          blocklength.
    """

    def __init__(self, W_single, n, verbose=False):
        self.tb = TypeBasedChannel(np.asarray(W_single, dtype=float), int(n),
                                   verbose=verbose)
        self.W_single = np.asarray(W_single, dtype=float)
        self.n = int(n)
        self.verbose = bool(verbose)

    # -- per-output-block staircase data (sort order is fixed, channel-only) ---
    def _blocks(self):
        tb = self.tb
        out = []
        for (s, e) in tb.cond_y_x.iterate_cond():
            alpha_blk = tb.alpha_coeffs[s:e]
            order = np.argsort(alpha_blk)[::-1]          # descending channel value
            nu = alpha_blk[order]                        # constants
            ridx = tb.R_to_Q[s:e][order]                 # Q-index per slab
            ratio = tb.R_Q_ratio[s:e][order]             # mass = Q[ridx]*ratio
            out.append((nu, ridx, ratio))
        return out

    # ------------------------------------------------------------------ QP ---
    def solve_rcu_plus(self, R):
        """
        Exact convex QP for the RCU+ kernel (`thm:qp`).

        Returns dict(Q_opt, P_e_exact, Gamma, status).
        inf_Q P_e = 1 - e^R * Gamma*.
        """
        tb = self.tb
        w0 = float(np.exp(-R))
        Q = cp.Variable(tb.num_q, nonneg=True)
        cons = [cp.sum(Q) == 1.0]
        obj_terms = []
        for (nu, ridx, ratio) in self._blocks():
            nb = len(nu)
            mass = cp.multiply(Q[ridx], ratio)           # linear in Q
            a = cp.Variable(nb, nonneg=True)
            cons.append(a[0] <= mass[0])
            for j in range(1, nb):
                cons.append(a[j] <= a[j - 1] + mass[j])
            cons.append(a <= w0)
            nu_ext = np.concatenate([nu, [0.0]])         # nu_{n+1}=0
            for j in range(nb):
                coef = float(nu_ext[j] - nu_ext[j + 1])  # nu_j - nu_{j+1} >= 0
                if coef <= 0.0:
                    continue
                # Abel/DCP-concave form so CLARABEL recognises concavity.
                obj_terms.append(coef * (w0 * a[j] - 0.5 * cp.square(a[j])))
        Gamma = cp.sum(obj_terms)
        prob = cp.Problem(cp.Maximize(Gamma), cons)
        prob.solve(solver=cp.CLARABEL)
        Gstar = float(Gamma.value)
        return {
            "Q_opt": np.asarray(Q.value, dtype=float),
            "P_e_exact": float(1.0 - np.exp(R) * Gstar),
            "Gamma": Gstar,
            "status": prob.status,
        }

    # ------------------------------------------------------------------ LP ---
    def solve_bracketing_lp(self, R, kernel="exact", K=64, side="both"):
        """
        General-kernel bracketing LP (`prop:pwl-lp`): PWL-approximate the fixed
        antiderivative Phi and bound inf_Q P_e from below (tangent) and above
        (secant).

        kernel : 'exact' (exact RC, no QP exists) or 'rcu' (for cross-check).
        Returns dict(P_lo, P_hi, gap, Q_lo, Q_hi).
        For the exact kernel: P_e = 1 - J, J = sum_y sum_j (nu_j-nu_{j+1}) Phi(sigma_j),
        Phi(t) = (1-(1-t)^M)/M with M = e^R + 1.
        """
        if kernel == "exact":
            M = float(np.exp(R) + 1.0)
            Phi = lambda t: _phi_exact(t, M)
            dPhi = lambda t: _dphi_exact(t, M)
            grid = np.linspace(0.0, 1.0, K + 1)
        elif kernel == "rcu":
            eR, w0 = float(np.exp(R)), float(np.exp(-R))
            def Phi(t):
                a = np.minimum(t, w0)
                return eR * (w0 * a - 0.5 * a * a)
            def dPhi(t):
                return np.where(t <= w0, 1.0 - eR * t, 0.0)
            grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, K + 1), [w0]]))
        else:
            raise ValueError(kernel)

        Phi_g = Phi(grid)
        dPhi_g = dPhi(grid)
        blocks = self._blocks()

        def _solve(which):
            tb = self.tb
            Q = cp.Variable(tb.num_q, nonneg=True)
            cons = [cp.sum(Q) == 1.0]
            obj_terms = []
            for (nu, ridx, ratio) in blocks:
                nb = len(nu)
                mass = cp.multiply(Q[ridx], ratio)
                # sigma_j = cumulative mass
                sig = []
                run = 0
                for j in range(nb):
                    run = mass[j] if j == 0 else run + mass[j]
                    sig.append(run)
                nu_ext = np.concatenate([nu, [0.0]])
                for j in range(nb):
                    coef = float(nu_ext[j] - nu_ext[j + 1])
                    if coef <= 0.0:
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
                    obj_terms.append(coef * z)
            J = cp.sum(obj_terms)
            prob = cp.Problem(cp.Maximize(J), cons)
            prob.solve(solver=cp.CLARABEL)
            return float(J.value), np.asarray(Q.value, dtype=float)

        J_sec, Q_sec = _solve("secant")    # Jhat^- <= J  -> upper bound on P_e
        J_tan, Q_tan = _solve("tangent")   # Jhat^+ >= J  -> lower bound on P_e
        P_hi = 1.0 - J_sec
        P_lo = 1.0 - J_tan
        return {"P_lo": P_lo, "P_hi": P_hi, "gap": P_hi - P_lo,
                "Q_lo": Q_tan, "Q_hi": Q_sec}
