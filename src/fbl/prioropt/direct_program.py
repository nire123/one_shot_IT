r"""
Direct convex program on the simplex (channel) -- the unified Phi-view.

Both the achievability and converse prior optimizations are the *same* program

        minimize   P_e(Q) = 1 - c^T Phi(A Q)        over  Q in the simplex,

where ``A Q`` are the cumulative per-output staircase masses ``sigma``, ``c`` the
metric gaps ``nu_j - nu_{j+1}``, and the **kernel chooses Phi**:

    * achievability (RCU+, L=1):  Phi(t) = t - 1/2 e^R t^2   (clamped at w0=e^-R)
                                  Phi'(t) = max(0, 1 - t/w0)        -> smooth (QP)
    * converse (Dirac kernel):    Phi(t) = min(t, w0)   (the ramp)
                                  Phi'(t) = 1{t < w0}               -> piecewise-linear (LP)

Both are concave in Q (Phi concave, sigma linear), so this is a smooth/PWL concave
maximization over the simplex.  We solve it *directly* with a first-order march
(projected gradient or Frank-Wolfe), using the analytic water-fill gradient

        g(x) = dGamma/dQ(x) = sum_y sum_{i fed by x} ratio_i * sum_{j>=i} c_j Phi'(sigma_j),

and the directional derivative on the simplex is  D_mu Gamma = <g - mean(g), mu>
(only the centered gradient acts, since sum_x mu(x)=0).  Optimality is the
water-filling condition: g(x) equal on the support.

STATUS / RECOMMENDATION.  This is the *unified* view and the scalable path
(cheap iterations, warm-startable, exact for any kernel -- no bracketing gap).
But first-order convergence is sublinear and stalls near machine precision
because Phi is flat past the clamp (not strongly concave), and the converse
(Dirac) is non-smooth and crawls. For exact high-precision single solves prefer
``AchievabilityQP`` (RCU+) and ``TypeBasedChannel.optimize_prior`` / the
meta-converse LP. A support-identifying active-set finish (future work) would
make the direct program competitive at high precision.
"""
import numpy as np

from fbl.prioropt.achievability_qp import AchievabilityQP


def _project_simplex(v):
    """Euclidean projection of v onto {x>=0, sum x = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - css / ind > 0
    rho = ind[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


class DirectPriorOpt:
    """
    Direct first-order prior optimization on the simplex (channel coding).

    Parameters
    ----------
    W_single : (k_x, k_y) channel
    n        : blocklength
    """

    def __init__(self, W_single, n):
        self._aqp = AchievabilityQP(np.asarray(W_single, float), int(n))
        self.blocks = self._aqp._blocks()          # (nu, ridx, ratio) per output block
        self.num_q = self._aqp.tb.num_q

    # ---- objective Gamma = c^T Phi(A Q) and its water-fill gradient ----------
    def _gamma_grad(self, Q, w0, kernel):
        G = 0.0
        grad = np.zeros(self.num_q)
        eR = 1.0 / w0
        for (nu, ridx, ratio) in self.blocks:
            sigma = np.cumsum(ratio * Q[ridx])
            if kernel == "rcu":                    # smooth clamped parabola
                Phi = np.where(sigma <= w0, sigma - 0.5 * eR * sigma ** 2, 0.5 * w0)
                Phip = np.maximum(0.0, 1.0 - eR * sigma)
            elif kernel in ("converse", "dirac"):  # the ramp
                Phi = np.minimum(sigma, w0)
                Phip = (sigma < w0).astype(float)
            else:
                raise ValueError(f"unknown kernel {kernel!r}")
            coef = nu - np.append(nu[1:], 0.0)     # nu_j - nu_{j+1} >= 0
            G += float(np.sum(coef * Phi))
            suffix = np.cumsum((coef * Phip)[::-1])[::-1]
            np.add.at(grad, ridx, ratio * suffix)
        return G, grad

    def directional_derivative(self, Q, mu, R, kernel="rcu"):
        """D_mu Gamma(Q) = <g, mu>; requires sum(mu)=0 to stay on the simplex."""
        w0 = float(np.exp(-R))
        _, g = self._gamma_grad(np.asarray(Q, float), w0, kernel)
        return float(g @ np.asarray(mu, float))

    # ---- solver --------------------------------------------------------------
    def solve(self, R, kernel="rcu", method="pgd", warm_start=None,
              max_iter=5000, tol=1e-9):
        """
        Minimize P_e(Q) = 1 - Gamma(Q) over the simplex by a first-order march.

        Parameters
        ----------
        R       : total rate (nats); threshold w0 = e^{-R}
        kernel  : 'rcu' (achievability) or 'converse'/'dirac'
        method  : 'pgd' (projected gradient, recommended) or 'fw' (Frank-Wolfe)
        warm_start : optional initial prior (e.g. a neighbouring rate's solution)
        tol     : stop when the FW optimality gap < tol

        Returns dict(P_e, Q_opt, gap, iters, method, kernel).
        """
        w0 = float(np.exp(-R))
        Q = (np.ones(self.num_q) / self.num_q if warm_start is None
             else np.asarray(warm_start, float).copy())
        eta = w0
        it = 0
        for k in range(max_iter):
            G, g = self._gamma_grad(Q, w0, kernel)
            it = k + 1
            gap = float(g.max() - g @ Q)           # FW (optimality) gap on Gamma
            if gap < tol:
                break
            if method == "fw":
                x = int(np.argmax(g))
                d = -Q.copy(); d[x] += 1.0
                # exact-ish line search (golden section) on Gamma(Q + s d), s in [0,1]
                a, b, gr = 0.0, 1.0, 0.6180339887
                c, e = b - gr * (b - a), a + gr * (b - a)
                fc = self._gamma_grad(Q + c * d, w0, kernel)[0]
                fe = self._gamma_grad(Q + e * d, w0, kernel)[0]
                for _ in range(25):
                    if fc < fe:
                        a, c, fc = c, e, fe
                        e = a + gr * (b - a); fe = self._gamma_grad(Q + e * d, w0, kernel)[0]
                    else:
                        b, e, fe = e, c, fc
                        c = b - gr * (b - a); fc = self._gamma_grad(Q + c * d, w0, kernel)[0]
                Q = Q + 0.5 * (a + b) * d
            else:                                  # projected gradient + backtracking
                e = eta
                for _ in range(40):
                    Qn = _project_simplex(Q + e * g)
                    Gn = self._gamma_grad(Qn, w0, kernel)[0]
                    if Gn >= G + 1e-4 * e * (g @ (Qn - Q)):
                        break
                    e *= 0.5
                Q = Qn
                eta = min(e * 2.0, 1e6)
        G, _ = self._gamma_grad(Q, w0, kernel)
        return {"P_e": 1.0 - G, "Q_opt": Q, "gap": gap, "iters": it,
                "method": method, "kernel": kernel}
