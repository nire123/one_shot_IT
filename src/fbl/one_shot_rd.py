"""
One-Shot Rate-Distortion
=========================

Achievable bound (random coding) and converse (LP prior optimisation)
for one-shot rate-distortion coding.

    bound = OneShotRD(P_X, d)
    curve = bound.compute_curve(Q_Y)              # A-curve
    D     = bound.theory(curve, M)                # achievable expected distortion
    mc    = bound.mc(Q_Y, M, num_trials=1000)     # Monte Carlo estimate
    res   = bound.validate(Q_Y, M)                # compare theory vs MC
    Q_opt, D_opt = bound.optimize_prior(M)        # converse / LP optimum

Author: Nir
"""

import numpy as np
import cvxpy as cp

from fbl.one_shot_base import OneShotBase
from fbl.F_curve import merge_piecewise_linear_curves, integrate_curve_rd_exact


# ── A-curve construction ───────────────────────────────────────────────────────

def _build_A_curve_for_source_symbol(x, P_X, Q_Y, d):
    """
    Build A-curve contribution for a single source symbol x.

    Sort reconstruction symbols by d[x,y] ascending, then take cumulative
    sums of Q_Y (knots) and d[x,y]*Q_Y[y]*P_X[x] (values).
    """
    order    = np.argsort(d[x, :])
    d_sorted = d[x, order]
    Q_sorted = Q_Y[order]

    knots  = np.concatenate([[0.0], np.cumsum(Q_sorted)])
    values = np.concatenate([[0.0], np.cumsum(d_sorted * Q_sorted * P_X[x])])
    return knots, values


def _build_A_curve(P_X, Q_Y, d):
    """
    Build complete A_X(w) curve across all source symbols.

    A_X(w) = E_{P_X x Q_Y}[d(X,Y) * 1{PEC(Y|X) <= w}]
    Merges per-source-symbol curves.
    """
    all_knots, all_values = zip(*[
        _build_A_curve_for_source_symbol(x, P_X, Q_Y, d)
        for x in range(len(P_X))
    ])
    return merge_piecewise_linear_curves(all_knots, all_values)


class OneShotRD(OneShotBase):
    """
    One-shot rate-distortion: achievable bound and LP prior optimisation.

    Parameters
    ----------
    P_X : np.ndarray, shape (X_size,)
        Source distribution.
    d : np.ndarray, shape (X_size, Y_size)
        Distortion matrix. d[x, y] = distortion between source symbol x
        and reconstruction symbol y.
    """

    def __init__(self, P_X, d):
        self.P_X    = np.asarray(P_X)
        self.d      = np.asarray(d)
        self.X_size = len(self.P_X)
        self.Y_size = self.d.shape[1]
        assert np.isclose(self.P_X.sum(), 1.0), f"P_X sums to {self.P_X.sum():.6f}"
        assert self.d.shape == (self.X_size, self.Y_size), \
            f"d shape {self.d.shape} != ({self.X_size}, {self.Y_size})"

    # ── abstract methods ───────────────────────────────────────────────────────

    def compute_curve(self, Q_Y):
        """
        Build A-curve from reconstruction prior Q_Y.

        Parameters
        ----------
        Q_Y : np.ndarray, shape (Y_size,)

        Returns
        -------
        (knots, A_vals) : tuple of np.ndarray
        """
        Q_Y = np.asarray(Q_Y)
        assert Q_Y.shape == (self.Y_size,), f"Q_Y shape {Q_Y.shape} != ({self.Y_size},)"
        assert np.isclose(Q_Y.sum(), 1.0), f"Q_Y sums to {Q_Y.sum():.6f}"
        return _build_A_curve(self.P_X, Q_Y, self.d)

    def draw_random_code(self, Q_Y, M, rng):
        """Draw M codewords (reconstruction symbols) i.i.d. from Q_Y."""
        return rng.choice(self.Y_size, size=M, p=Q_Y)

    def evaluate(self, code):
        """
        Exact expected distortion for one codebook.

        Each source symbol x is encoded to the nearest codeword under d[x, :].
        Result is averaged over source distribution P_X.
        """
        return float(sum(
            self.P_X[x] * self.d[x, code].min()
            for x in range(self.X_size)
        ))

    def theory(self, curve, M, num_refined_points=1000):
        """Integrate A-curve to get theoretical expected distortion."""
        knots, vals = curve
        return integrate_curve_rd_exact(knots, vals, M, num_refined_points)

    def optimize_prior(self, M):
        """
        Solve LP to find the reconstruction prior Q_Y that minimises distortion.

        LP:
            min  sum_{x,y} P_X[x] * d[x,y] * R[x,y]
            s.t. R[x,y] <= Q_Y[y]        for all x,y  (R bounded by prior)
                 sum_y R[x,y] >= 1/M     for all x    (coverage per source symbol)
                 sum_y Q_Y[y] = 1
                 R >= 0,  Q_Y >= 0

        Parameters
        ----------
        M : int or float

        Returns
        -------
        prior  : np.ndarray, shape (Y_size,), or None
        metric : float (expected distortion), or None
        """
        R   = cp.Variable((self.X_size, self.Y_size), nonneg=True)
        Q_Y = cp.Variable((self.Y_size, 1),           nonneg=True)

        c_per_source = cp.sum(R, axis=1) >= 1.0 / M   # named so we can read dual

        cost      = self.P_X.reshape(-1, 1) * self.d
        objective = cp.Minimize(cp.sum(cp.multiply(cost, R)))
        constraints = [
            R            <= Q_Y.T,    # R[x,y] <= Q_Y[y], broadcast over x
            c_per_source,             # sum_y R[x,y] >= 1/M  for each x
            cp.sum(Q_Y)  == 1.0,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            self._s_dual = None
            return None, None

        self._s_dual = np.abs(c_per_source.dual_value.flatten())
        prior  = Q_Y.value.flatten()
        metric = float(problem.value)
        return prior, metric

    def check_kkt(self, M, Q_Y, s):
        """
        Verify the two KKT optimality conditions for rate-distortion.

        With w = 1/M fixed, define for each reconstruction symbol y:

            h_y(s) = sum_x min(s_x, P_X[x] * d[x,y])

        Condition 1 — prior optimality:
            Q_Y[y] > 0  =>  h_y(s) = min_{y'} h_{y'}(s)
            Q_Y[y] = 0  =>  h_y(s) >= min_{y'} h_{y'}(s)

        Condition 2 — source coverage (per source symbol x):
            Q_Y{y : P_X[x]*d[x,y] < s_x}  <=  1/M  <=  Q_Y{y : P_X[x]*d[x,y] <= s_x}

        Parameters
        ----------
        M   : int or float
        Q_Y : np.ndarray, shape (Y_size,)   optimal prior
        s   : np.ndarray, shape (X_size,)   dual variables from optimize_prior

        Returns
        -------
        dict with keys:
            cond1       bool
            cond2       bool
            all_pass    bool
            h           np.ndarray  - h_y values for each y
            cond2_slack np.ndarray  - (mass_below, mass_leq) pairs per x
        """
        w = 1.0 / M

        # ── Condition 1 ────────────────────────────────────────────────────────
        # cost[x,y] = P_X[x] * d[x,y]; compare each column to s elementwise
        cost = self.P_X[:, None] * self.d          # (X_size, Y_size)
        h = np.array([
            np.sum(np.minimum(s, cost[:, y]))
            for y in range(self.Y_size)
        ])
        h_min   = h.min()
        support = Q_Y > 1e-6   # LP noise threshold
        cond1 = (
            np.all(np.abs(h[support]  - h_min) < 1e-5) and
            np.all(        h[~support] - h_min > -1e-5)
        )

        # ── Condition 2 ────────────────────────────────────────────────────────
        # Use tolerance around s_x for the same floating-point reason as channel.
        tol = 1e-6
        cond2_slack = np.array([
            (Q_Y[cost[x, :] <  s[x] - tol].sum(),   # mass strictly below s_x
             Q_Y[cost[x, :] <= s[x] + tol].sum())    # mass at or below s_x
            for x in range(self.X_size)
        ])
        mass_below = cond2_slack[:, 0]
        mass_leq   = cond2_slack[:, 1]
        cond2 = bool(
            np.all(mass_below <= w + 1e-5) and
            np.all(mass_leq   >= w - 1e-5)
        )

        return {
            'cond1':       cond1,
            'cond2':       cond2,
            'all_pass':    cond1 and cond2,
            'h':           h,
            'cond2_slack': cond2_slack,
        }
