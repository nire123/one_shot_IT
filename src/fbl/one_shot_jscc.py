"""
One-Shot JSCC
=============

Standalone one-shot implementation of joint source-channel coding (JSCC).

Problem setup
-------------
- Source  : V ~ P_V,  alphabet size |V|
- Encoder : f: V -> X  (deterministic, drawn randomly)
- Channel : W[x,y] = P(Y=y | X=x)
- Decoder : MAP rule  v_hat = argmax_v  P_V[v] * W[f(v), y]

The key metric for each (v, x, y) triple is:

    alpha[v, x, y] = |V| * P_V[v] * W[x, y]

and the F-curve is the CDF of alpha under the joint distribution
    Q_V(v) * Q_{X|V}(x|v) * W(y|x)

where Q_V = uniform (1/|V|) following the standard JSCC LP formulation.

Bounds
------
converse (LP lower bound on error):
    Solve the meta-converse LP to get an upper bound on success probability.
    Returns  error_lb = 1 - LP_value.

achievable (RCB upper bound on error):
    error_ub = M * integral_0^{1/M}  F(w) dw
    where F is the JSCC F-curve under the chosen encoder distribution Q_{X|V}.
    M is typically |V|^n (= |V| for n=1).

Monte Carlo:
    Draw random encoders  f(v) ~ Q_{X|V}(.|v) independently for each v,
    evaluate exact error via MAP decoder, average over trials.

Author: Nir
"""

import numpy as np
import cvxpy as cp

from fbl.F_curve import integrate_curve_jscc, merge_piecewise_linear_curves


class OneShotJSCC:

    def __init__(self, P_V: np.ndarray, W: np.ndarray):
        """
        Parameters
        ----------
        P_V : array, shape (|V|,)
            Source distribution.
        W   : array, shape (|X|, |Y|)
            Channel transition matrix, W[x, y] = P(Y=y | X=x).
        """
        self.P_V   = np.asarray(P_V, dtype=float)
        self.W     = np.asarray(W,   dtype=float)
        self.v_size = len(P_V)
        self.x_size, self.y_size = W.shape

        # alpha[v, x, y] = |V| * P_V[v] * W[x, y]
        self._alpha = (self.v_size
                       * self.P_V[:, None, None]
                       * self.W[None, :, :])  # (v, x, y)

    # ── converse ───────────────────────────────────────────────────────────────

    def compute_converse(self, M: float):
        """
        Meta-converse LP: upper bound on success probability for codebook size M.

        Variables : R(v,x,y) >= 0,  Q(v,x) >= 0
        Objective : max  sum_{v,x,y}  alpha[v,x,y] * R[v,x,y]
        Subject to:
            R[v,x,y]             <= (1/|V|) * Q[v,x]          for all v,x,y
            sum_{v,x} R[v,x,y]  <= 1/M                        for all y
            sum_x    Q[v,x]      == 1                          for all v

        Returns
        -------
        error_lb : float
            Lower bound on error probability  (= 1 - LP value).
        Q_XgV    : ndarray, shape (|V|, |X|)
            Optimal encoder distribution  Q_{X|V}(x|v).
            None if LP did not solve to optimality.
        """
        v, x, y = self.v_size, self.x_size, self.y_size

        R = cp.Variable((v, x, y), nonneg=True)
        Q = cp.Variable((v, x),    nonneg=True)

        c_per_output = cp.sum(R, axis=(0, 1)) <= 1.0 / M   # dual → s[y]

        constraints = [
            R               <= (1.0 / v) * Q[:, :, None],
            c_per_output,
            cp.sum(Q, axis=1)      == 1.0,
        ]

        prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(self._alpha, R))),
                          constraints)
        prob.solve(solver=cp.CLARABEL)

        if prob.status not in ('optimal', 'optimal_inaccurate'):
            return None, None

        # Store dual variables for check_kkt
        self._s_dual = np.abs(c_per_output.dual_value.flatten())  # shape (y_size,)

        return 1.0 - float(prob.value), Q.value

    def check_kkt(self, M: float, Q_XgV: np.ndarray, s: np.ndarray):
        """
        Verify KKT optimality conditions for the JSCC converse LP solution.

        For a fixed dual threshold vector s[y] >= 0 and w = 1/M, define

            g_{v,x}(s) = sum_y  min(alpha[v,x,y], s[y])  -  w * sum_y s[y]

        where  alpha[v,x,y] = |V| * P_V[v] * W[x,y].

        Condition 1 — encoder optimality (per source symbol v):
            For each v, the support of Q[v,:] must achieve the minimum of
            g_{v,x}(s) over x, and all x outside the support must be >= that min.

                Q[v,x] > 0  =>  g_{v,x}(s) = min_{x'} g_{v,x'}(s)
                Q[v,x] = 0  =>  g_{v,x}(s) >= min_{x'} g_{v,x'}(s)

        Condition 2 — threshold optimality (per output y):
            sum_{(v,x): alpha[v,x,y] > s[y]}  Q[v,x]/|V|  <=  1/M
            sum_{(v,x): alpha[v,x,y] >= s[y]} Q[v,x]/|V|  >=  1/M

        Parameters
        ----------
        M     : float     codebook size
        Q_XgV : ndarray, shape (|V|, |X|)   optimal encoder distribution
        s     : ndarray, shape (|Y|,)        dual variables from compute_converse

        Returns
        -------
        dict with keys:
            cond1       : bool   condition 1 holds for every source symbol v
            cond2       : bool   condition 2 holds for every output y
            all_pass    : bool
            g           : ndarray shape (|V|, |X|)   g_{v,x} values
            cond2_slack : ndarray shape (|Y|, 2)      (mass_above, mass_geq) per y
        """
        Q = np.asarray(Q_XgV, dtype=float)
        s = np.asarray(s,     dtype=float)
        w = 1.0 / M
        tol = 1e-6

        # g[v, x] = sum_y min(alpha[v,x,y], s[y]) - w * sum(s)
        # alpha shape: (v, x, y);  s shape: (y,)
        g = (np.minimum(self._alpha, s[None, None, :]).sum(axis=2)
             - w * s.sum())                                # (v, x)

        # ── Condition 1 ────────────────────────────────────────────────────────
        cond1 = True
        for vi in range(self.v_size):
            g_v      = g[vi]              # (x,)
            q_v      = Q[vi]              # (x,)
            g_min    = g_v.min()
            support  = q_v > tol
            ok = (np.all(np.abs(g_v[support]  - g_min) < 1e-5) and
                  np.all(       g_v[~support] - g_min  > -1e-5))
            if not ok:
                cond1 = False
                break

        # ── Condition 2 ────────────────────────────────────────────────────────
        # weight[v, x] = Q[v, x] / |V|
        weight = Q / self.v_size   # (v, x)

        cond2_slack = np.empty((self.y_size, 2))
        for yi in range(self.y_size):
            alpha_y = self._alpha[:, :, yi]   # (v, x)
            above   = alpha_y >  s[yi] + tol
            geq     = alpha_y >= s[yi] - tol
            cond2_slack[yi, 0] = weight[above].sum()   # mass strictly above
            cond2_slack[yi, 1] = weight[geq  ].sum()   # mass at-or-above

        mass_above = cond2_slack[:, 0]
        mass_geq   = cond2_slack[:, 1]
        cond2 = bool(np.all(mass_above <= w + 1e-5) and
                     np.all(mass_geq   >= w - 1e-5))

        return {
            'cond1':       cond1,
            'cond2':       cond2,
            'all_pass':    cond1 and cond2,
            'g':           g,
            'cond2_slack': cond2_slack,
        }

    # ── F-curve ────────────────────────────────────────────────────────────────

    def compute_f_curve(self, Q_XgV: np.ndarray):
        """
        Build the JSCC F-curve.

        For each output symbol y the curve is built in the DESCENDING-alpha
        representation used by the RCB bound:

          x-axis (knots)  : cumulative encoder probability P(alpha_y >= threshold)
          y-axis (vals)   : cumulative partial expectation
                            E[alpha_y * 1{alpha_y >= threshold}] / total

        where total = sum_{v,x,y} weight[v,x] * alpha[v,x,y]  ~= 1.

        The per-y curves are merged by interpolating onto shared knots and summing.

        The RCB bound then uses:  M * integral_0^{1/M}  (1 - F(w))  dw

        Parameters
        ----------
        Q_XgV : array, shape (|V|, |X|)
            Encoder distribution  Q_{X|V}(x|v).

        Returns
        -------
        knots : ndarray in [0, 1]
        vals  : ndarray in [0, 1]
        """
        Q_XgV  = np.asarray(Q_XgV, dtype=float)
        weight = Q_XgV / self.v_size          # (v, x), sums to 1

        per_knots = []
        per_vals  = []
        rtol, atol = 1e-3, 1e-5

        for y in range(self.y_size):
            alpha_y = self._alpha[:, :, y]    # (v, x)
            flat_a  = alpha_y.ravel().copy()
            flat_w  = weight.ravel().copy()

            # sort descending by alpha
            order  = np.argsort(flat_a)[::-1]
            flat_a = flat_a[order]
            flat_w = flat_w[order]

            # merge entries with close alpha values
            not_close = np.concatenate(
                [[True], ~np.isclose(flat_a[:-1], flat_a[1:], rtol=rtol, atol=atol)])
            gid   = np.cumsum(not_close) - 1
            new_w = np.bincount(gid, weights=flat_w)
            _, li = np.unique(gid, return_index=True)
            new_a = flat_a[li]

            # cumulative sums (prepend 0)
            cs_w  = np.insert(np.cumsum(new_w),       0, 0.0)
            cs_wa = np.insert(np.cumsum(new_w * new_a), 0, 0.0)

            # deduplicate consecutive close entries in cs_w
            not_close2 = np.concatenate(
                [[True], ~np.isclose(cs_w[:-1], cs_w[1:], rtol=rtol, atol=atol)])
            gid2   = np.cumsum(not_close2) - 1
            _, li2 = np.unique(gid2, return_index=True)
            cs_w  = cs_w[li2]
            cs_wa = cs_wa[li2]

            per_knots.append(cs_w)
            per_vals.append(cs_wa)

        # total (should be ≈ 1); normalise before merging
        s = sum(v[-1] for v in per_vals)

        merged_knots, merged_vals = merge_piecewise_linear_curves(
            per_knots, [v / s for v in per_vals])

        return merged_knots, merged_vals

    # ── achievable bound ───────────────────────────────────────────────────────

    def achievable_bound(self, M: float, Q_XgV: np.ndarray):
        """
        RCB upper bound on error probability for codebook size M.

            error_ub = M * integral_0^{1/M}  F(w) dw

        Parameters
        ----------
        M     : float   codebook size (|V|^n for blocklength n)
        Q_XgV : array, shape (|V|, |X|)

        Returns
        -------
        float
        """
        knots, vals = self.compute_f_curve(Q_XgV)
        return integrate_curve_jscc(knots, vals, M)

    # ── random code ────────────────────────────────────────────────────────────

    def draw_random_code(self, Q_XgV: np.ndarray, rng: np.random.Generator):
        """
        Draw a random encoder  f: V -> X  with  f(v) ~ Q_{X|V}(.|v).

        Returns
        -------
        codebook : ndarray, shape (|V|,)
            codebook[v] = x(v)
        """
        Q_XgV = np.asarray(Q_XgV, dtype=float)
        return np.array([rng.choice(self.x_size, p=Q_XgV[v])
                         for v in range(self.v_size)])

    def evaluate(self, codebook: np.ndarray):
        """
        Exact error probability for a fixed encoder using the MAP decoder.

        MAP decoder: v_hat(y) = argmax_v  P_V[v] * W[codebook[v], y]

        Parameters
        ----------
        codebook : array, shape (|V|,)   codebook[v] = x(v)

        Returns
        -------
        float
        """
        codebook = np.asarray(codebook, dtype=int)

        # scores[v, y] = P_V[v] * W[x(v), y]
        scores  = self.P_V[:, None] * self.W[codebook, :]  # (v, y)
        decoded = np.argmax(scores, axis=0)                 # (y,) best v for each y

        error = 0.0
        for v in range(self.v_size):
            x = codebook[v]
            wrong_y = decoded != v
            error  += self.P_V[v] * self.W[x, wrong_y].sum()

        return float(error)

    # ── Monte Carlo ────────────────────────────────────────────────────────────

    def mc(self, Q_XgV: np.ndarray, num_trials: int = 1000, seed=None):
        """
        Monte Carlo estimate of the random-coding error probability.

        Each trial draws an independent random encoder via draw_random_code
        and evaluates it exactly via evaluate.

        Parameters
        ----------
        Q_XgV     : array, shape (|V|, |X|)
        num_trials : int
        seed      : int or None   base seed (trial t uses seed+t)

        Returns
        -------
        dict with keys: mean, std, trials
        """
        Q_XgV = np.asarray(Q_XgV, dtype=float)
        results = []
        for trial in range(num_trials):
            rng  = np.random.default_rng(None if seed is None else seed + trial)
            code = self.draw_random_code(Q_XgV, rng)
            results.append(self.evaluate(code))

        arr = np.array(results)
        return {
            'mean':   float(arr.mean()),
            'std':    float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            'trials': arr,
        }
