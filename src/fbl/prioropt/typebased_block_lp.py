r"""
Type-based block LP for the prior-optimised random-coding bound.

Implements the kink-adaptive / block-LP scheme from `p1po_rcu_draft.tex`
(PHD repo) at the type-based / blocklength-n level. Built on top of the
existing TypeBasedChannel single-threshold LP machinery.

Idea
----
At a fixed threshold w (equivalently rate R = -log w), the reverse-channel
identity expresses the success spectrum as

    S(Q; w) = sup_{V}  sum_{x,y} V(x|y) W(y|x)
              s.t.     0 <= V(x|y) <= Q(x),  sum_x V(x|y) = w.

The type-based LP `TypeBasedChannel.compute_opt_Q(M)` is exactly this sup
with w = 1/M, lifted to the type-based variables (Q over input types,
R over (output-type, conditional-type) pairs):

    cap:        R[r] <= Q[T_X(r)] * (1/w) * R_Q_ratio[r]
    proper:     sum_{r in y-block} R[r] <= 1
    objective:  sum_r R[r] * alpha[r] * w  =  S(Q; w).

The block LP for K thresholds 0 < w_1 < ... < w_K replicates the (cap,
proper) constraints once per threshold, with the SHARED prior Q, and
maximises

    sum_{k=1..K} c_k * w_k * sum_r R^(k)[r] * alpha[r]
       =  sum_k c_k * S(Q; w_k),

i.e.\ a chord-rule estimate of int_0^{w_K} S(Q; w) dw with chord-to-zero
(S(Q; 0) = 0).  Because S is concave in w (Lemma `p1porcu:lem-waterfill`),
the chord lies *below* S, so

    sum_k c_k S(Q; w_k)  <=  int_0^{w_K} S(Q; w) dw,

and therefore

    P^ub(Q) := 1 - (1/w_K) * sum_k c_k S(Q; w_k)
            >= P(R; Q),   with R = -log(w_K).

Solving the block LP gives an optimal type prior Q_opt and an upper
bound P^ub on the random-coding bound at Q_opt; the true random-coding
bound P(R; Q_opt) is then evaluated separately via the F-curve.
"""

import os
import sys


import numpy as np
import cvxpy as cp

from fbl.channel_achievable_type_based import TypeBasedChannel


def chord_weights_to_zero(w_grid):
    """
    Trapezoidal chord-rule weights for int_0^{w_K} f(w) dw given values
    at 0 < w_1 < ... < w_K and the boundary value f(0) = 0.

    Returns c in R^K with int_chord = sum_k c_k * f(w_k).
    """
    w = np.asarray(w_grid, dtype=float)
    K = len(w)
    if K == 1:
        return np.array([w[0] / 2.0])
    c = np.empty(K)
    # Trapezoidal sum is sum_{k=1..K} (w_k - w_{k-1})/2 * (f_{k-1} + f_k)
    # with w_0 = 0 and f_0 = 0.  Expanding gives, for 1 <= k < K,
    #   coefficient of f_k = (w_{k+1} - w_{k-1}) / 2   (using w_0 = 0)
    # and for k = K, coefficient = (w_K - w_{K-1}) / 2.
    c[0] = w[1] / 2.0                       # = (w_2 - 0) / 2
    for k in range(1, K - 1):
        c[k] = (w[k + 1] - w[k - 1]) / 2.0
    c[K - 1] = (w[K - 1] - w[K - 2]) / 2.0
    return c


class TypeBasedBlockLP:
    """
    Multi-threshold block LP wrapping a TypeBasedChannel.

    Parameters
    ----------
    W_single : (k_x, k_y) ndarray
        Single-letter channel.
    n : int
        Blocklength.
    verbose : bool
        Forwarded to the underlying solver and TypeBasedChannel.
    """

    def __init__(self, W_single, n, verbose=False):
        self.tb = TypeBasedChannel(W_single, n, verbose=verbose)
        self.W_single = np.asarray(W_single)
        self.n = int(n)
        self.verbose = bool(verbose)

    def solve(self, w_grid, chord_weights=None):
        """
        Solve the block LP at the K ascending thresholds in `w_grid`.

        Parameters
        ----------
        w_grid : (K,) ndarray
            Thresholds 0 < w_1 < ... < w_K, ascending.
        chord_weights : (K,) ndarray or None
            Chord-rule weights.  If None, the chord-to-zero trapezoidal
            weights are used (the canonical choice for chord-rule
            integration of S with S(0) = 0).

        Returns
        -------
        dict with keys
            Q_opt           (num_q,)  optimal type prior
            S_per_k         (K,)      S(Q_opt; w_k) at each threshold
            chord_integral  float     sum_k c_k * S(Q_opt; w_k)
            P_ub            float     1 - (1/w_K) * chord_integral
            lp_value        float     LP objective at the optimum
            lp_status       str       solver status
        """
        w = np.asarray(w_grid, dtype=float)
        K = len(w)
        assert K >= 1, "need at least one threshold"
        assert np.all(np.diff(w) > 0), "w_grid must be strictly ascending"
        assert w[0] > 0 and w[-1] <= 1, "thresholds must lie in (0, 1]"

        c = (chord_weights_to_zero(w) if chord_weights is None
             else np.asarray(chord_weights, dtype=float))
        assert len(c) == K

        Q_var = cp.Variable((self.tb.num_q,), nonneg=True)
        R_vars = [cp.Variable((self.tb.num_R,), nonneg=True) for _ in range(K)]

        constraints = [cp.sum(Q_var) == 1.0]
        for k in range(K):
            # Reverse-channel cap at threshold w_k:
            #   R^(k)[r] <= Q[T_X(r)] * (1/w_k) * R_Q_ratio[r]
            constraints.append(
                R_vars[k] <= cp.multiply(
                    Q_var[self.tb.R_to_Q],
                    (1.0 / w[k]) * self.tb.R_Q_ratio,
                )
            )
            # Proper-conditional cap, per output-type block:
            for st, ed in self.tb.cond_y_x.iterate_cond():
                constraints.append(cp.sum(R_vars[k][st:ed]) <= 1.0)

        # Objective: sum_k c_k * w_k * sum_r R^(k)[r] * alpha[r]
        obj_terms = [
            c[k] * w[k] * cp.sum(cp.multiply(R_vars[k], self.tb.alpha_coeffs))
            for k in range(K)
        ]
        problem = cp.Problem(cp.Maximize(sum(obj_terms)), constraints)
        problem.solve(solver=cp.SCIPY,
                      scipy_options={"method": "highs-ds"},
                      verbose=self.verbose)

        Q_opt = np.asarray(Q_var.value, dtype=float)
        S_per_k = np.array([
            float(w[k] * np.sum(R_vars[k].value * self.tb.alpha_coeffs))
            for k in range(K)
        ])
        chord_integral = float(np.sum(c * S_per_k))
        P_ub = 1.0 - (1.0 / w[K - 1]) * chord_integral

        return {
            "Q_opt": Q_opt,
            "S_per_k": S_per_k,
            "chord_integral": chord_integral,
            "P_ub": float(P_ub),
            "lp_value": float(problem.value),
            "lp_status": problem.status,
        }


def rcu_plus_from_F_curve(knots, F_repo, w_max):
    """
    Evaluate the prior-side RCU+ bound

        P(R; Q) = 1 - e^R * int_0^{e^{-R}} S(Q; w) dw

    given the repo's F-curve (knots, F_repo).  In the repo's convention,
    F_repo on `knots` is the success spectrum S(Q; w) (piecewise-linear,
    starting at 0 and ending at 1).  Set w_max = e^{-R} = 1/M.

    Implementation: piecewise-linear interpolation of F_repo on a fine
    grid in [0, w_max], then trapezoidal integration.

    Returns
    -------
    P : float
        P(R; Q) = 1 - (1/w_max) * int_0^{w_max} S(w) dw.
    """
    knots = np.asarray(knots, dtype=float)
    F = np.asarray(F_repo, dtype=float)
    grid = np.linspace(0.0, w_max, 5001)
    F_at_grid = np.interp(grid, knots, F)
    integral = np.trapz(F_at_grid, grid)
    return float(1.0 - (1.0 / w_max) * integral)


def marginal_input(Q_T, n, k_x):
    """
    Compute the per-symbol marginal P(x) induced by a type prior P_T_X.

    P(x) = E_{T ~ P_T_X}[T[x] / n].
    """
    from fbl.type_class_core import enumerate_type_class
    marg = np.zeros(k_x)
    for i_T, T in enumerate(enumerate_type_class(n, k_x)):
        if Q_T[i_T] <= 0:
            continue
        marg += Q_T[i_T] * (T / n)
    return marg
