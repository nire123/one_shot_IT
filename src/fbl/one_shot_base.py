"""
One-Shot Base Class
====================

Abstract base class for one-shot channel coding and rate-distortion.
Covers both the achievable bound (random coding) and the converse
(prior optimisation via LP).

The flow for any concrete subclass:

    bound = OneShotChannel(W)   # or OneShotRD(P_X, d)

    curve  = bound.compute_curve(prior)              # F-curve or A-curve
    code   = bound.draw_random_code(prior, M, rng)   # one random codebook
    metric = bound.evaluate(code)                    # exact error / distortion
    metric = bound.theory(curve, M)                  # theoretical value

mc() and validate() are implemented once here using these four methods.

Author: Nir
"""

import numpy as np
from abc import ABC, abstractmethod


class OneShotBase(ABC):

    # ── abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def compute_curve(self, prior):
        """
        Build the curve (knots, vals) from the prior.

        Parameters
        ----------
        prior : np.ndarray
            Distribution used to draw random codes.
            Channel coding: Q over input alphabet X.
            Rate-distortion: Q_Y over reconstruction alphabet Y.

        Returns
        -------
        (knots, vals) : tuple of np.ndarray
        """

    @abstractmethod
    def draw_random_code(self, prior, M, rng):
        """
        Draw M codewords i.i.d. from prior.

        Parameters
        ----------
        prior : np.ndarray
        M : int
            Codebook size.
        rng : np.random.Generator

        Returns
        -------
        code : np.ndarray, shape (M,)
            Indices into the alphabet.
        """

    @abstractmethod
    def evaluate(self, code):
        """
        Compute exact metric (error probability or distortion) for one codebook.

        Parameters
        ----------
        code : np.ndarray, shape (M,)
            Codebook drawn by draw_random_code.

        Returns
        -------
        float
        """

    @abstractmethod
    def theory(self, curve, M, num_refined_points=1000):
        """
        Integrate curve to get theoretical metric.

        Parameters
        ----------
        curve : tuple (knots, vals)
            Output of compute_curve.
        M : int
            Codebook size.
        num_refined_points : int

        Returns
        -------
        float
        """

    @abstractmethod
    def check_kkt(self, M, prior, s):
        """
        Verify KKT optimality conditions for the result of optimize_prior.

        Parameters
        ----------
        M     : int or float
        prior : np.ndarray   optimal prior returned by optimize_prior
        s     : np.ndarray   dual variables stored in self._s_dual after optimize_prior

        Returns
        -------
        dict with at minimum:
            cond1    : bool
            cond2    : bool
            all_pass : bool
        """

    @abstractmethod
    def optimize_prior(self, M):
        """
        Solve LP to find the prior that minimises the metric for codebook size M.

        Parameters
        ----------
        M : int or float
            Codebook size (LP accepts real-valued M).

        Returns
        -------
        prior : np.ndarray or None
            Optimal prior distribution. None if LP failed.
        metric : float or None
            Achieved metric (error probability or distortion). None if LP failed.
        """

    # ── shared implementation ──────────────────────────────────────────────────

    def mc(self, prior, M, num_trials=1000, seed=None):
        """
        Monte Carlo estimate: average metric over random codebooks.

        Each trial draws an independent codebook via draw_random_code,
        then evaluates it via evaluate.

        Parameters
        ----------
        prior : np.ndarray
        M : int
        num_trials : int
        seed : int, optional
            Base seed. Trial t uses seed+t, giving independent reproducible trials.

        Returns
        -------
        dict with keys: mean, std, trials
        """
        results = []
        for trial in range(num_trials):
            rng = np.random.default_rng(None if seed is None else seed + trial)
            code = self.draw_random_code(prior, M, rng)
            results.append(self.evaluate(code))

        arr = np.array(results)
        return {
            'mean':   float(arr.mean()),
            'std':    float(arr.std(ddof=1)),
            'trials': arr,
        }

    def validate(self, prior, M, num_trials=500, seed=42, num_refined_points=1000):
        """
        Compare theory vs Monte Carlo within a 95% confidence interval.

        Parameters
        ----------
        prior : np.ndarray
        M : int
        num_trials : int
        seed : int
        num_refined_points : int

        Returns
        -------
        dict with keys:
            theory, mc_mean, mc_std, mc_margin,
            lower_ci, upper_ci, within_ci, M
        """
        curve      = self.compute_curve(prior)
        theory_val = self.theory(curve, M, num_refined_points)
        mc_res     = self.mc(prior, M, num_trials, seed)

        margin = 1.96 * mc_res['std'] / np.sqrt(num_trials)
        lower  = mc_res['mean'] - margin
        upper  = mc_res['mean'] + margin

        return {
            'theory':    theory_val,
            'mc_mean':   mc_res['mean'],
            'mc_std':    mc_res['std'],
            'mc_margin': margin,
            'lower_ci':  lower,
            'upper_ci':  upper,
            'within_ci': lower <= theory_val <= upper,
            'M':         M,
        }
