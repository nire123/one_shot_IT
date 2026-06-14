"""
Type-Based Rate-Distortion
===========================

Type-based achievable bound and LP prior optimisation for memoryless sources.

    tb = TypeBasedRD(P_X_single, d_single, n)

    # A-curve for a type prior
    curve = tb.compute_curve(P_T_Y)
    D     = tb.theory(curve, M)

    # LP optimal type prior
    P_T_opt, D_opt = tb.optimize_prior(M)

    # Equivalent n-letter one-shot object (for validation)
    one_shot = tb.get_one_shot_object()   # OneShotRD(P_X^n, d^n)

Author: Nir
"""

import numpy as np

from fbl.type_based_base import TypeBasedBase
from fbl.one_shot_rd import OneShotRD
from fbl.F_curve import integrate_curve_rd_exact
from fbl.type_class_core import composition_count
from fbl.type_based_utils import type_prior_to_one_shot

# Core type-based engine from the existing implementation
from fbl.rd_achievable_type_based import TypeBasedRateDistortion as _Core


class TypeBasedRD(TypeBasedBase):
    """
    Type-based rate-distortion: achievable bound and LP prior optimisation.

    Works with type priors P_T_Y over reconstruction types instead of sequence
    priors, reducing complexity from O(k^n) to O(n^k).

    Parameters
    ----------
    P_X_single : np.ndarray, shape (k_x,)
        Single-letter source distribution.
    d_single : np.ndarray, shape (k_x, k_y)
        Single-letter distortion matrix.
    n : int
        Blocklength.
    """

    def __init__(self, P_X_single, d_single, n):
        self.P_X_single = np.asarray(P_X_single, dtype=float)
        self.d_single   = np.asarray(d_single,   dtype=float)
        self.n          = n
        self.k_x        = len(self.P_X_single)
        self.k_y        = self.d_single.shape[1]
        self.num_types  = composition_count(n, self.k_y)

        assert np.isclose(self.P_X_single.sum(), 1.0), \
            "P_X_single must sum to 1"
        assert self.d_single.shape == (self.k_x, self.k_y), \
            f"d_single shape {self.d_single.shape} != ({self.k_x}, {self.k_y})"

        # Build the core type-enumeration engine once
        self._core = _Core(self.P_X_single, self.d_single, n)

    # ── abstract methods ───────────────────────────────────────────────────────

    def get_one_shot_object(self):
        """
        Return OneShotRD for the n-letter product source/distortion.

        This is the reference object used to validate the type-based results.
        """
        P_X_n = self.P_X_single.copy()
        d_n   = self.d_single.copy()
        for _ in range(self.n - 1):
            P_X_n = np.kron(P_X_n, self.P_X_single)
            tmp   = d_n[:, None, :, None] + self.d_single[None, :, None, :]
            d_n   = tmp.reshape(tmp.shape[0] * tmp.shape[1], tmp.shape[2] * tmp.shape[3])
        return OneShotRD(P_X_n, d_n)

    def compute_curve(self, P_T_Y):
        """
        Build A-curve from type prior P_T_Y.

        Parameters
        ----------
        P_T_Y : np.ndarray, shape (num_types,)

        Returns
        -------
        (knots, A_vals) : tuple of np.ndarray
        """
        P_T_Y = np.asarray(P_T_Y)
        assert P_T_Y.shape == (self.num_types,), \
            f"P_T_Y shape {P_T_Y.shape} != ({self.num_types},)"
        assert np.isclose(P_T_Y.sum(), 1.0), f"P_T_Y sums to {P_T_Y.sum():.6f}"
        return self._core.build_A_curve_type_based(P_T_Y)

    def theory(self, curve, M, num_refined_points=1000):
        """Integrate A-curve to get theoretical expected distortion."""
        knots, vals = curve
        return integrate_curve_rd_exact(knots, vals, M, num_refined_points)

    def optimize_prior(self, M):
        """
        Solve type-based LP to find optimal type prior.

        The LP minimises expected distortion over type-constant priors.
        At n=1 this is equivalent to the one-shot LP.
        For n>1 it optimises over a restricted (type-constant) subset.

        Parameters
        ----------
        M : int or float

        Returns
        -------
        P_T_opt : np.ndarray, shape (num_types,), or None
        metric  : float (expected distortion), or None
        """
        distortion = self._core.compute_opt_Q(M)
        if distortion is None:
            return None, None
        return self._core.Q_values.copy(), float(distortion)

    # ── convenience ───────────────────────────────────────────────────────────

    def prior_to_one_shot(self, P_T_Y):
        """Convert type prior P_T_Y to the equivalent sequence prior Q over Y^n."""
        return type_prior_to_one_shot(np.asarray(P_T_Y), self.n, self.k_y)
