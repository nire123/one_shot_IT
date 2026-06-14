"""
Type-Based Channel Coding
==========================

Type-based achievable bound and LP prior optimisation for memoryless channels.

    tb = TypeBasedChannel(W_single, n)

    # F-curve for a type prior
    curve = tb.compute_curve(P_T_X)
    P_err = tb.theory(curve, M)

    # LP optimal type prior
    P_T_opt, P_err_opt = tb.optimize_prior(M)

    # Equivalent n-letter one-shot object (for validation)
    one_shot = tb.get_one_shot_object()   # OneShotChannel(W^⊗n)

Author: Nir
"""

import numpy as np

from fbl.type_based_base import TypeBasedBase
from fbl.one_shot_channel import OneShotChannel
from fbl.F_curve import integrate_curve_channel_coding_exact
from fbl.type_class_core import composition_count
from fbl.type_based_utils import type_prior_to_one_shot
from fbl.channel_achievable_utils import kronecker_power

# Core type-based engine from the existing implementation
from fbl.channel_achievable_type_based import TypeBasedChannel as _Core


class TypeBasedChannel(TypeBasedBase):
    """
    Type-based channel coding: achievable bound and LP prior optimisation.

    Works with type priors P_T_X over input types instead of sequence priors,
    reducing complexity from O(k^n) to O(n^k).

    Parameters
    ----------
    W_single : np.ndarray, shape (k_x, k_y)
        Single-letter channel transition matrix.  Each row must sum to 1.
    n : int
        Blocklength.
    """

    def __init__(self, W_single, n):
        self.W_single = np.asarray(W_single)
        self.n        = n
        self.k_x, self.k_y = self.W_single.shape
        self.num_types = composition_count(n, self.k_x)

        assert np.allclose(self.W_single.sum(axis=1), 1.0), \
            "W_single rows must sum to 1"

        # Build the core type-enumeration engine once
        self._core = _Core(W_single, n)

    # ── abstract methods ───────────────────────────────────────────────────────

    def get_one_shot_object(self):
        """
        Return OneShotChannel for the n-letter product channel W^⊗n.

        This is the reference object used to validate the type-based results.
        """
        W_n = kronecker_power(self.W_single, self.n)
        return OneShotChannel(W_n)

    def compute_curve(self, P_T_X):
        """
        Build F-curve from type prior P_T_X.

        Parameters
        ----------
        P_T_X : np.ndarray, shape (num_types,)

        Returns
        -------
        (knots, F_vals) : tuple of np.ndarray
        """
        P_T_X = np.asarray(P_T_X)
        assert P_T_X.shape == (self.num_types,), \
            f"P_T_X shape {P_T_X.shape} != ({self.num_types},)"
        assert np.isclose(P_T_X.sum(), 1.0), f"P_T_X sums to {P_T_X.sum():.6f}"
        return self._core.build_F_curve_type_based(P_T_X)

    def theory(self, curve, M, num_refined_points=1000):
        """Integrate F-curve to get theoretical error probability."""
        knots, vals = curve
        return integrate_curve_channel_coding_exact(knots, vals, M, num_refined_points)

    def optimize_prior(self, M):
        """
        Solve type-based LP to find optimal type prior.

        The LP maximises success probability over type-constant priors.
        At n=1 this is equivalent to the one-shot LP.
        For n>1 it optimises over a restricted (type-constant) subset.

        Parameters
        ----------
        M : int or float

        Returns
        -------
        P_T_opt : np.ndarray, shape (num_types,), or None
        metric  : float (error probability), or None
        """
        error = self._core.compute_opt_Q(M)
        if error is None:
            return None, None
        return self._core.Q_values.copy(), float(error)

    # ── convenience ───────────────────────────────────────────────────────────

    def prior_to_one_shot(self, P_T_X):
        """Convert type prior P_T_X to the equivalent sequence prior Q over X^n."""
        return type_prior_to_one_shot(np.asarray(P_T_X), self.n, self.k_x)
