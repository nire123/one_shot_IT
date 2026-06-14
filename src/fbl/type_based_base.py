"""
Type-Based One-Shot Base Class
================================

Abstract base class for type-based channel coding and rate-distortion.

The type-based approach exploits the memoryless structure of the n-letter
problem.  Instead of working with all k^n sequences, it works with the
polynomial-sized set of types (empirical distributions).

Compared to OneShotBase:
  - prior is a type prior P_T  (shape: num_types)  instead of a sequence prior
  - no mc / validate  (validation is done by comparing against OneShotBase)
  - adds get_one_shot_object() to build the equivalent n-letter one-shot object

Author: Nir
"""

from abc import ABC, abstractmethod


class TypeBasedBase(ABC):

    # ── abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def get_one_shot_object(self):
        """
        Build and return the equivalent one-shot object for the n-letter problem.

        For channel coding: returns OneShotChannel(W^⊗n).
        For rate-distortion: returns OneShotRD(P_X^n, d^⊗n).

        Used to validate the type-based implementation against the one-shot one.
        """

    @abstractmethod
    def compute_curve(self, prior):
        """
        Build curve (knots, vals) from a type prior.

        Parameters
        ----------
        prior : np.ndarray, shape (num_types,)
            Distribution over types.  num_types = composition_count(n, k).

        Returns
        -------
        (knots, vals) : tuple of np.ndarray
        """

    @abstractmethod
    def theory(self, curve, M, num_refined_points=1000):
        """
        Integrate curve to get theoretical metric (error prob or distortion).

        Parameters
        ----------
        curve : tuple (knots, vals)
        M     : int or float   codebook size
        num_refined_points : int

        Returns
        -------
        float
        """

    @abstractmethod
    def optimize_prior(self, M):
        """
        Solve type-based LP to find the type prior that optimises the metric.

        Parameters
        ----------
        M : int or float

        Returns
        -------
        prior  : np.ndarray, shape (num_types,), or None
        metric : float, or None
        """
