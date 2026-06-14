"""
Type-Based One-Shot JSCC
========================

Type-based implementation of joint source-channel coding (JSCC).

Problem setup (blocklength n)
------------------------------
- Source  : V^n ~ P_V^n,   single-letter alphabet size k_v
- Encoder : f: V^n -> X^n  random, type-constant
- Channel : W^n,            single-letter W[x,y] = P(Y=y|X=x)
- Decoder : MAP rule

Type-constant encoder
---------------------
A type-constant encoder assigns probability to each (V^n, X^n) pair
based only on their joint empirical type T_{VX} (a k_v × k_x count matrix):

    Q(x^n | v^n) = P_T_VX[ index(T_{VX}) ]  /  |T_{VX|V}|

where P_T_VX is a distribution over joint type indices in cond_vx.

For n=1 every sequence is its own type so there is an exact bijection with
the one-shot encoder Q_{X|V}:

    P_T_VX[ v * k_x + x ]  =  Q_{X|V}[v, x] / k_v

F-curve
-------
For each output type T_Y the Lorenz-style curve is built in descending-alpha
order, then all per-output-type curves are merged.

Converse LP
-----------
Analogous to the one-shot LP but counting in terms of types.

Author: Nir
"""

import numpy as np
import cvxpy as cp

from fbl.type_class_core import (
    get_joint_source_channel_types,
    conditional_enum,
    composition_count,
    log_size_type_class,
)
from fbl.F_curve import merge_piecewise_linear_curves


class TypeBasedJSCC:
    """
    Type-based JSCC achievable bound and converse for blocklength n.

    Parameters
    ----------
    P_V   : array, shape (k_v,)   single-letter source distribution
    W     : array, shape (k_x, k_y)  single-letter channel
    n     : int                   blocklength
    """

    def __init__(self, P_V: np.ndarray, W: np.ndarray, n: int):
        self.P_V   = np.asarray(P_V, dtype=float)
        self.W     = np.asarray(W,   dtype=float)
        self.n     = n
        self.k_v   = len(P_V)
        self.k_x, self.k_y = W.shape

        log_W   = np.log(np.clip(W,   1e-300, None))
        log_P_V = np.log(np.clip(P_V, 1e-300, None))

        (self._log_alpha,
         self._log_ratio,
         self._T_vx_idx,      # maps cond_y_vx entry -> cond_vx entry
         self._cond_y_vx,     # conditional_enum(n, k_y, k_v*k_x)
         self._cond_vx,       # conditional_enum(n, k_v, k_x)
         ) = get_joint_source_channel_types(
             n, self.k_v, self.k_x, self.k_y, log_W, log_P_V)

        self._alpha = np.exp(self._log_alpha)
        self._ratio = np.exp(self._log_ratio)

        # Number of joint (T_V, T_{X|V}) types  =  cond_vx.len
        self.num_vx_types = self._cond_vx.len

    # ── helpers ────────────────────────────────────────────────────────────────

    def q_xgv_to_type_prior(self, Q_XgV: np.ndarray) -> np.ndarray:
        """
        Convert one-shot Q_{X|V}  (shape k_v × k_x, rows sum to 1)
        to a type-based prior P_T_VX  (shape cond_vx.len).

        Valid only for n=1; raises AssertionError otherwise.
        """
        assert self.n == 1, "q_xgv_to_type_prior is only valid for n=1"
        Q = np.asarray(Q_XgV, dtype=float)
        assert Q.shape == (self.k_v, self.k_x)
        # Delegate to memoryless_prior which handles indexing correctly.
        return self.memoryless_prior(Q)

    # ── F-curve ────────────────────────────────────────────────────────────────

    def compute_f_curve(self, P_T_VX: np.ndarray):
        """
        Build the JSCC F-curve for a given type-based encoder prior.

        Parameters
        ----------
        P_T_VX : array, shape (cond_vx.len,)
            Distribution over joint (T_V, T_{X|V}) types.  Must sum to 1.

        Returns
        -------
        knots : ndarray in [0, 1]
        vals  : ndarray in [0, 1]
        """
        P = np.asarray(P_T_VX, dtype=float)
        assert len(P) == self.num_vx_types, (
            f"P_T_VX length {len(P)} != cond_vx.len {self.num_vx_types}")

        # weight[r] = P_T_VX[ T_vx-index of r ] * ratio[r]
        weight = P[self._T_vx_idx] * self._ratio

        all_knots = []
        all_vals  = []

        for st, ed in self._cond_y_vx.iterate_cond():
            a = self._alpha[st:ed]
            w = weight[st:ed]

            # sort descending by alpha
            order = np.argsort(a)[::-1]
            a_s   = a[order]
            w_s   = w[order]

            knots = np.concatenate([[0.0], np.cumsum(w_s)])
            vals  = np.concatenate([[0.0], np.cumsum(w_s * a_s)])

            all_knots.append(knots)
            all_vals.append(vals)

        # total (should be ≈ k_v^n for a normalised P_T_VX)
        s = sum(v[-1] for v in all_vals)

        # normalise each curve, then merge
        merged_knots, merged_vals = merge_piecewise_linear_curves(
            all_knots, [v / s for v in all_vals])

        return merged_knots, merged_vals

    # ── achievable bound ───────────────────────────────────────────────────────

    def achievable_bound(self, M: float, P_T_VX: np.ndarray) -> float:
        """
        RCB upper bound on error probability:   M * ∫_0^{1/M} (1-F(w)) dw.
        """
        from fbl.F_curve import integrate_curve_jscc
        knots, vals = self.compute_f_curve(P_T_VX)
        return integrate_curve_jscc(knots, vals, M)

    # ── converse LP ────────────────────────────────────────────────────────────

    def compute_converse(self, M: float):
        """
        Type-based meta-converse LP via CVXPY.

        Variables : R (num_R,) >= 0,  Q (num_Q,) >= 0
        Maximise  : alpha @ R
        Subject to:
            R[r] <= ratio[r] * Q[T_vx(r)]          for all r
            Σ_{r in T_Y block} R[r] <= k_v^n / M   for each T_Y block
            Σ_{r in T_V block} Q[r] == 1            for each T_V block

        Returns
        -------
        error_lb : float   1 - LP_value
        Q_cond   : ndarray, shape (cond_vx.len,)
                   optimal Q, each T_V block sums to 1
        """
        num_R = self._cond_y_vx.len
        num_Q = self._cond_vx.len
        kv_n  = float(self.k_v ** self.n)

        R = cp.Variable(num_R, nonneg=True)
        Q = cp.Variable(num_Q, nonneg=True)

        constraints = [R <= cp.multiply(Q[self._T_vx_idx], self._ratio)]
        for st, ed in self._cond_y_vx.iterate_cond():
            constraints.append(cp.sum(R[st:ed]) <= kv_n / M)
        for st, ed in self._cond_vx.iterate_cond():
            constraints.append(cp.sum(Q[st:ed]) == 1.0)

        prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(self._alpha, R))),
                          constraints)
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            return None, None

        return 1.0 - float(prob.value), Q.value

    def memoryless_prior(self, Q_XgV: np.ndarray) -> np.ndarray:
        """
        Compute the type-based prior P_T_VX induced by the memoryless encoder
        Q_{X|V}^n coupled with the source P_V^n.

        P[T_VX] ∝ (n! / ∏_{v,x} t_{vx}!) × ∏_{v,x} (P_V(v)·Q(x|v))^{t_{vx}}

        The result is scaled by 1/k_v^n so that knots in compute_f_curve
        stay in [0, 1] (same convention as expected by achievable_bound).

        Parameters
        ----------
        Q_XgV : array, shape (k_v, k_x), rows sum to 1

        Returns
        -------
        P_T_VX : array, shape (cond_vx.len,)
        """
        Q   = np.asarray(Q_XgV, dtype=float)
        log_Q  = np.log(np.clip(Q, 1e-300, None))
        # Reference distribution over V is uniform (P_V is already encoded
        # in alpha values; including it here would double-count it).
        log_joint = log_Q   # shape (k_v, k_x): only Q(x|v), no P_V

        from fbl.type_class_core import log_size_conditional_type_class
        P = np.zeros(self.num_vx_types)
        for i_T_v, T_v, i_T_x_given_v, T_vx, _ in self._cond_vx.enumerate():
            # T_vx is a (k_v, k_x) count matrix (row sums = T_v).
            # Use conditional size (product of row multinomials), NOT joint size.
            # This ensures each T_V block sums to 1/k_v^n as required.
            log_p = log_size_conditional_type_class(T_vx) + np.sum(T_vx * log_joint)
            idx   = self._cond_vx.tuple_2_ix(i_T_v, i_T_x_given_v)
            P[idx] = np.exp(log_p)

        return P / float(self.k_v ** self.n)

    # ── one-shot bridge ────────────────────────────────────────────────────────

    def get_one_shot(self):
        """Return the corresponding OneShotJSCC object (same P_V, W)."""
        from fbl.one_shot_jscc import OneShotJSCC
        return OneShotJSCC(self.P_V, self.W)
