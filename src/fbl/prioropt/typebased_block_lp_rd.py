r"""
Type-based block LP for the RD prior-optimised random-coding bound.

Kernel
------
The RD random-coding bound is

    D(Q_Y) = M(M-1) int_0^1 A_X(Q_Y; w) (1-w)^{M-2} dw,

where A_X(Q_Y; w) is the type-based A-curve (built by
TypeBasedRateDistortion.build_A_curve_type_based).  The kernel
K_M(w) = M(M-1)(1-w)^{M-2} on [0, 1] is the Beta(1, M-1) density
times M; in z = -log w space it is bell-shaped (Gumbel-like for large M)
centred at z* = log(M-1).

A_X is *convex* in w (the slopes are sorted distortion values, ascending).
The chord of a convex function lies *above* the function, so chord-rule
integration of A_X against the (positive) kernel *overestimates* the
integral -- exactly what we want for an achievability (upper) bound on D.

Block LP
--------
Variables: type prior Q over Y-types and one reverse-channel block
R^(k) per threshold w_k.  Each block at threshold w_k realises

    A_X(Q; w_k) = w_k * sum_r R^(k)[r] * d_coeffs[r]

via cap and coverage constraints identical in shape to the existing
single-threshold RD LP (TypeBasedRateDistortion.compute_opt_Q).

Chord-rule + right-tail upper bound on D:

    D(Q) <= sum_k c_k * A_X(Q; w_k)  +  tail_coef * A_X(Q; 1),
    c_k       = int_0^{w_K} phi_k(w) K_M(w) dw         (hat at w_k)
    tail_coef = int_{w_K}^1 K_M(w) dw  =  M (1 - w_K)^{M-1}

A_X(Q; 1) = E_{P_X x Q_Y}[d] is linear in Q (precomputed as `d_marg`).
The objective is linear in (Q, R^(k)); the LP is convex.
"""

import os
import sys


import numpy as np
import cvxpy as cp

from fbl.rd_achievable_type_based import TypeBasedRateDistortion
from fbl.type_class_core import enumerate_type_class
from fbl.F_curve import integrate_curve_rd_exact


# ---------------------------------------------------------------------- grids


def grid_around_zstar(z_star, delta, K, z_min=0.05):
    """
    K thresholds in z, equally spaced around z_star with step delta.

    Offsets are (-(K-1)/2, ..., (K-1)/2) * delta.  Values are clipped at
    z_min so we never approach w = 1 too closely; duplicates from clipping
    are dropped.  Returned in ascending z order.
    """
    offsets = (np.arange(K) - (K - 1) / 2.0) * delta
    z = z_star + offsets
    z = np.maximum(z, z_min)
    z = np.unique(np.sort(z))
    return z


# ----------------------------------------------------------- chord coefficients


def chord_weights_rd(w_grid, M, num_grid_points=10000):
    """
    Kernel-weighted hat-function chord coefficients per the chapter
    formulation `p1porcu:eq-c`.  The user supplies interior grid points
    w_1 < ... < w_K in (0, 1); the grid is implicitly augmented with
    w_0 = 0 (anchor F(Q;0)=0, so c_0 multiplies zero and is not returned)
    and w_{K+1} = 1 (anchor F(Q;1) = E_{P_X x Q_Y}[d], linear in Q).

    Each interior coefficient is the kernel-weighted integral of the
    piecewise-linear hat function centred at w_k:

        c_k = int_{w_{k-1}}^{w_k} (w - w_{k-1})/(w_k - w_{k-1}) * K_M(w) dw
            + int_{w_k}^{w_{k+1}} (w_{k+1} - w)/(w_{k+1} - w_k) * K_M(w) dw

    (with w_{K+1} = 1 in the right piece of c_K so phi_K extends fully
    to w = 1).  The endpoint coefficient at w = 1 is

        c_at_1 = int_{w_K}^1 (w - w_K)/(1 - w_K) * K_M(w) dw.

    Mass identity: c_K's right piece + c_at_1 = int_{w_K}^1 K_M dw =
    M(1 - w_K)^{M-1}; the chapter splits this kernel mass between the
    F(Q; w_K) and F(Q; 1) endpoints, instead of lumping all of it onto
    a constant-F(1) tail bound.  For convex F with F(w_K) < F(1) the
    chord interpolant lies strictly below the constant F(1) on [w_K, 1],
    so this rule is uniformly tighter.

    Returns
    -------
    c       : (K,) ndarray of chord-rule coefficients on user's grid.
    c_at_1  : float, coefficient on F(Q; 1).
    """
    w = np.asarray(w_grid, dtype=float)
    assert np.all(np.diff(w) > 0), "w_grid must be strictly ascending"
    assert w[0] > 0 and w[-1] < 1, "w_grid must lie strictly in (0, 1)"
    K = len(w)

    # Augmented grid: w_full = [0, w_1, ..., w_K, 1]
    w_full = np.concatenate([[0.0], w, [1.0]])

    def Km(ww):
        return M * (M - 1) * (1.0 - ww) ** (M - 2)

    def integrate_hat(left, centre, right):
        """int [phi_{centre}(w)] * K_M(w) dw  with phi piecewise linear,
        0 at left and right, 1 at centre."""
        out = 0.0
        if centre - left > 1e-15:
            ww = np.linspace(left, centre, num_grid_points)
            hat = (ww - left) / (centre - left)
            out += float(np.trapz(hat * Km(ww), ww))
        if right - centre > 1e-15:
            ww = np.linspace(centre, right, num_grid_points)
            hat = (right - ww) / (right - centre)
            out += float(np.trapz(hat * Km(ww), ww))
        return out

    c = np.empty(K)
    for k in range(K):
        c[k] = integrate_hat(w_full[k], w_full[k + 1], w_full[k + 2])

    # Endpoint at w = 1: left piece on [w_K, 1], no right piece.
    if 1.0 - w[K - 1] > 1e-15:
        ww = np.linspace(w[K - 1], 1.0, num_grid_points)
        hat = (ww - w[K - 1]) / (1.0 - w[K - 1])
        c_at_1 = float(np.trapz(hat * Km(ww), ww))
    else:
        c_at_1 = 0.0
    return c, c_at_1


# Backwards-compatible alias (was: chord-to-zero left + constant-tail right).
# Kept here only so old call sites surface in tests if they exist.
def chord_weights_rd_with_tail(w_grid, M, num_grid_points=10000):
    raise RuntimeError(
        "chord_weights_rd_with_tail has been replaced by chord_weights_rd "
        "(chapter formulation p1porcu:eq-c).  Update call sites."
    )


# ------------------------------------------------------------------ block LP


class TypeBasedBlockLPRD:
    """
    Multi-threshold block LP for RD achievability.

    Parameters
    ----------
    P_X_single : (k_x,) ndarray, source distribution
    d_single   : (k_x, k_y) ndarray, distortion matrix
    n          : int, blocklength
    verbose    : bool
    """

    def __init__(self, P_X_single, d_single, n, verbose=False):
        self.tb = TypeBasedRateDistortion(P_X_single, d_single, n,
                                          verbose=verbose)
        self.P_X_single = np.asarray(P_X_single, dtype=float)
        self.d_single = np.asarray(d_single, dtype=float)
        self.n = int(n)
        self.verbose = bool(verbose)

        # Precompute d_marg[i_T_Y] = sum_y T_Y[y] * E_{P_X}[d(X, y)]
        # so that A_X(P_T_Y; 1) = sum_{T_Y} P_T_Y(T_Y) * d_marg(T_Y).
        #
        # Units: d_coeffs in TypeBasedRateDistortion is the TOTAL (n-letter)
        # joint distortion summed over the joint type, times P_X_prob[T_X].
        # The chord-rule contributions w_k * sum_r R^(k)[r] * d_coeffs[r]
        # are therefore in n-letter total-distortion units, so d_marg must
        # be too (no division by n) to make A_X(Q; 1) match.
        k_x, k_y = self.d_single.shape
        EdY = self.P_X_single @ self.d_single   # (k_y,) -> E[d(X, y)]
        num_q = self.tb.num_q
        d_marg = np.empty(num_q)
        for i_T_Y, T_Y in enumerate(enumerate_type_class(n, k_y)):
            d_marg[i_T_Y] = float((T_Y * EdY).sum())
        self.d_marg = d_marg

    def solve(self, w_grid, M, chord_coefs=None, c_at_1=None):
        """
        Solve the block LP at thresholds w_grid for codebook size M.

        Parameters
        ----------
        w_grid : (K,) ndarray, ascending in (0, 1).  Interior points only.
        M      : float, codebook size (drives chord weights & endpoint).
        chord_coefs, c_at_1 : optional precomputed kernel weights from
            chord_weights_rd(w_grid, M).

        Returns
        -------
        dict
            Q_opt      : (num_q,) optimal type prior
            A_per_k    : (K,) A_X(Q_opt; w_k)
            A_at_1     : float, A_X(Q_opt; 1) = E_{P_X x Q_Y}[d]
            chord_part : float, sum_k c_k * A_per_k
            tail_part  : float, c_at_1 * A_at_1  (kernel-weighted endpoint)
            D_ub       : float, chord_part + tail_part  (LP objective)
            lp_status  : solver status
        """
        w = np.asarray(w_grid, dtype=float)
        K = len(w)
        if chord_coefs is None or c_at_1 is None:
            c, c1 = chord_weights_rd(w, M)
        else:
            c = np.asarray(chord_coefs, dtype=float)
            c1 = float(c_at_1)

        tb = self.tb
        Q_var = cp.Variable((tb.num_q,), nonneg=True)
        R_vars = [cp.Variable((tb.num_R,), nonneg=True) for _ in range(K)]

        constraints = [cp.sum(Q_var) == 1.0]
        for k in range(K):
            # Cap: R^(k)[r] <= Q[T_Y(r)] * (1/w_k) * R_Q_ratio[r]
            constraints.append(
                R_vars[k] <= cp.multiply(
                    Q_var[tb.R_to_Q],
                    (1.0 / w[k]) * tb.R_Q_ratio,
                )
            )
            # Coverage per source type
            for st, ed in tb.cond_x_y.iterate_cond():
                constraints.append(cp.sum(R_vars[k][st:ed]) >= 1.0)

        # Objective:
        #   sum_k c_k * w_k * sum_r R^(k)[r] * d_coeffs[r]      (chord interior)
        # + c_at_1 * sum_q Q[q] * d_marg[q]                       (endpoint at 1)
        obj_terms = [
            c[k] * w[k] * cp.sum(cp.multiply(R_vars[k], tb.d_coeffs))
            for k in range(K)
        ]
        obj_terms.append(c1 * cp.sum(cp.multiply(Q_var, self.d_marg)))
        problem = cp.Problem(cp.Minimize(sum(obj_terms)), constraints)
        try:
            problem.solve(solver=cp.SCIPY,
                          scipy_options={"method": "highs-ds"},
                          verbose=self.verbose)
        except Exception:
            problem.solve(solver=cp.CLARABEL, verbose=self.verbose)

        Q_opt = np.asarray(Q_var.value, dtype=float)
        A_per_k = np.array([
            float(w[k] * np.sum(R_vars[k].value * tb.d_coeffs))
            for k in range(K)
        ])
        A_at_1 = float(np.sum(Q_opt * self.d_marg))
        chord_part = float(np.sum(c * A_per_k))
        tail_part = float(c1 * A_at_1)
        return {
            "Q_opt": Q_opt,
            "A_per_k": A_per_k,
            "A_at_1": A_at_1,
            "chord_part": chord_part,
            "tail_part": tail_part,
            "D_ub": chord_part + tail_part,
            "lp_value": float(problem.value),
            "lp_status": problem.status,
        }


# --------------------------------------------------------------- evaluator


def true_D_at_Q(tb, P_T_Y, M, num_refined_points=2000):
    """
    True random-coding RD bound at type prior P_T_Y:
        D = M(M-1) * int_0^1 A_X(w) (1-w)^{M-2} dw,
    using the exact F_curve integrator on the type-based A-curve.
    """
    knots, A = tb.build_A_curve_type_based(P_T_Y)
    return integrate_curve_rd_exact(knots, A, M, num_refined_points=num_refined_points)


def marginal_y(P_T_Y, n, k_y):
    """Per-symbol marginal Q_Y(y) induced by the type prior."""
    out = np.zeros(k_y)
    for i_T, T in enumerate(enumerate_type_class(n, k_y)):
        if P_T_Y[i_T] <= 0:
            continue
        out += P_T_Y[i_T] * (T / n)
    return out
