"""
One-Shot Channel Coding
========================

Achievable bound (random coding) and converse (LP prior optimisation)
for one-shot channel coding.

    bound = OneShotChannel(W)
    curve = bound.compute_curve(Q)          # F-curve
    P_err = bound.theory(curve, M)          # achievable error probability
    mc    = bound.mc(Q, M, num_trials=1000) # Monte Carlo estimate
    res   = bound.validate(Q, M)            # compare theory vs MC
    Q_opt, P_err_opt = bound.optimize_prior(M)   # converse / LP optimum

Author: Nir
"""

import numpy as np
import cvxpy as cp

from fbl.one_shot_base import OneShotBase
from fbl.F_curve import merge_piecewise_linear_curves, integrate_curve_channel_coding_exact


# ── F-curve construction ───────────────────────────────────────────────────────

def _build_F_curve_for_output(y, W, Q):
    """
    Build F-curve contribution for a single output symbol y.

    Sort inputs by W[x,y] descending, then take cumulative sums of Q
    (knots) and W[x,y]*Q[x] (values).
    """
    alpha_sorted = W[:, y][np.argsort(W[:, y])[::-1]]
    Q_sorted     = Q[np.argsort(W[:, y])[::-1]]

    knots  = np.concatenate([[0.0], np.cumsum(Q_sorted)])
    values = np.concatenate([[0.0], np.cumsum(alpha_sorted * Q_sorted)])
    return knots, values


def _build_F_curve(W, Q):
    """
    Build complete F-curve from channel W and prior Q.

    F(q) = success probability using the best q-fraction of inputs.
    Merges per-output curves and verifies F(1) = 1.
    """
    all_knots, all_values = zip(*[
        _build_F_curve_for_output(y, W, Q) for y in range(W.shape[1])
    ])
    merged_knots, merged_F = merge_piecewise_linear_curves(all_knots, all_values)

    assert np.isclose(merged_F[-1], 1.0, atol=1e-3), \
        f"F-curve final value {merged_F[-1]:.6f} != 1"

    return merged_knots, merged_F


class OneShotChannel(OneShotBase):
    """
    One-shot channel coding: achievable bound and LP prior optimisation.

    Parameters
    ----------
    W : np.ndarray, shape (X_size, Y_size)
        Channel transition matrix. W[x, y] = P(Y=y | X=x).
        Each row must sum to 1.
    """

    def __init__(self, W):
        self.W = np.array(W)
        self.X_size, self.Y_size = self.W.shape
        assert np.allclose(self.W.sum(axis=1), 1.0), "W rows must sum to 1"

    # ── abstract methods ───────────────────────────────────────────────────────

    def compute_curve(self, Q):
        """
        Build F-curve from prior Q over input alphabet.

        Parameters
        ----------
        Q : np.ndarray, shape (X_size,)

        Returns
        -------
        (knots, F_vals) : tuple of np.ndarray
        """
        Q = np.asarray(Q)
        assert Q.shape == (self.X_size,), f"Q shape {Q.shape} != ({self.X_size},)"
        assert np.isclose(Q.sum(), 1.0), f"Q sums to {Q.sum():.6f}"
        return _build_F_curve(self.W, Q)

    def draw_random_code(self, Q, M, rng):
        """Draw M codewords (input symbols) i.i.d. from Q."""
        return rng.choice(self.X_size, size=M, p=Q)

    def evaluate(self, code):
        """
        Exact error probability for one codebook under ML decoding.

        For each message m, error = sum of W[x_m, y] over outputs y
        where the ML decoder picks a different message.
        Averaged uniformly over messages.
        """
        M = len(code)
        scores    = self.W[code, :]                    # (M, Y_size)
        decoded_m = np.argmax(scores, axis=0)          # (Y_size,)

        err = sum(
            np.sum(self.W[code[m], decoded_m != m])
            for m in range(M)
        )
        return err / M

    def theory(self, curve, M, num_refined_points=1000):
        """Integrate F-curve to get theoretical error probability."""
        knots, vals = curve
        return integrate_curve_channel_coding_exact(knots, vals, M, num_refined_points)

    def optimize_prior(self, M):
        """
        Solve LP to find the input prior Q that minimises error probability.

        LP (equivalent to maximising success probability):
            max  sum_{x,y} W[x,y] * R[x,y]
            s.t. R[x,y] <= Q[x]          for all x,y  (R bounded by prior)
                 sum_x R[x,y] <= 1/M     for all y    (normalisation per output)
                 sum_x Q[x] = 1
                 R >= 0,  Q >= 0

        The dual variable s[y] for the per-output constraint encodes the
        optimal threshold at each output y (see check_kkt).

        Parameters
        ----------
        M : int or float

        Returns
        -------
        prior  : np.ndarray, shape (X_size,), or None
        metric : float (error probability), or None

        Side effect
        -----------
        self._s_dual : np.ndarray, shape (Y_size,)
            Dual variables for the per-output constraints; used by check_kkt.
        """
        R   = cp.Variable((self.X_size, self.Y_size), nonneg=True)
        Q   = cp.Variable((self.X_size, 1),           nonneg=True)

        c_per_output = cp.sum(R, axis=0) <= 1.0 / M   # named so we can read dual

        objective   = cp.Maximize(cp.sum(cp.multiply(self.W, R)))
        constraints = [
            R            <= Q,       # R[x,y] <= Q[x], broadcast over y
            c_per_output,            # sum_x R[x,y] <= 1/M  for each y
            cp.sum(Q)    == 1.0,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCIPY, scipy_options={"method": "highs-ds"}, verbose=False)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            self._s_dual = None
            return None, None

        self._s_dual = np.abs(c_per_output.dual_value.flatten())  # should be >= 0
        prior  = Q.value.flatten()
        metric = 1.0 - float(problem.value)   # error = 1 - success
        return prior, metric

    def check_kkt(self, M, Q, s):
        """
        Verify the two KKT optimality conditions (thesis Proposition).

        With w = 1/M fixed, define for each input x:

            g_x(s) = sum_y min(W[x,y], s_y)  -  w * sum_y s_y

        Condition 1 — input optimality:
            Q[x] > 0  =>  g_x(s) = min_{x'} g_{x'}(s)
            Q[x] = 0  =>  g_x(s) >= min_{x'} g_{x'}(s)

        Condition 2 — threshold optimality (per output y):
            Q{x : W[x,y] > s_y}  <=  w  <=  Q{x : W[x,y] >= s_y}

        Parameters
        ----------
        M : int or float
        Q : np.ndarray, shape (X_size,)   optimal prior
        s : np.ndarray, shape (Y_size,)   dual variables from optimize_prior

        Returns
        -------
        dict with keys:
            cond1       bool  - condition 1 holds
            cond2       bool  - condition 2 holds for every y
            all_pass    bool  - both conditions hold
            g           np.ndarray  - g_x values for each x
            cond2_slack np.ndarray  - (mass_above, mass_geq) pairs per y
        """
        w = 1.0 / M

        # ── Condition 1 ────────────────────────────────────────────────────────
        g = np.array([
            np.sum(np.minimum(self.W[x], s)) - w * s.sum()
            for x in range(self.X_size)
        ])
        g_min    = g.min()
        support  = Q > 1e-6   # LP solver can return ~1e-8 noise for true zeros
        cond1 = (
            np.all(np.abs(g[support]  - g_min) < 1e-5) and   # support equal
            np.all(        g[~support] - g_min > -1e-5)        # non-support >=
        )

        # ── Condition 2 ────────────────────────────────────────────────────────
        # Use tolerance: s[y] from the LP solver may land within floating-point
        # precision of a channel value W[x,y].  We therefore bracket the
        # threshold: "strictly above" excludes a tol-wide band, "at or above"
        # includes it.
        tol = 1e-6
        cond2_slack = np.array([
            (Q[self.W[:, y] >  s[y] + tol].sum(),   # mass strictly above s_y
             Q[self.W[:, y] >= s[y] - tol].sum())    # mass at or above s_y
            for y in range(self.Y_size)
        ])
        mass_above = cond2_slack[:, 0]
        mass_geq   = cond2_slack[:, 1]
        cond2 = bool(
            np.all(mass_above <= w + 1e-5) and
            np.all(mass_geq   >= w - 1e-5)
        )

        return {
            'cond1':       cond1,
            'cond2':       cond2,
            'all_pass':    cond1 and cond2,
            'g':           g,
            'cond2_slack': cond2_slack,
        }
