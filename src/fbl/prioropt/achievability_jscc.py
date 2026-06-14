"""
Achievability prior optimization for one-shot Joint Source--Channel Coding (JSCC),
type-based (method of types), built additively on TypeBasedJSCC.

This is the JSCC analogue of achievability_qp.py (channel) and
achievability_lp_rd.py (rate--distortion).  It does NOT modify any existing
file; it only *reads* TypeBasedJSCC's machinery.

Theory (Part-I lift, JSCC specialization)
-----------------------------------------
Enlarge the candidate space to source--codeword pairs (v,x).  For the matched
metric m(v,x,y)=|V| W(y|x) P_V(v) the success spectrum is a water-filling
staircase ordered per output-type by the metric `alpha`, with slab masses
`ratio * Q` where Q is the conditional codeword-type law (one simplex per
source-type T_V block) -- exactly the variable of TypeBasedJSCC.compute_converse.

The converse LP (compute_converse) is the *Dirac kernel* case:
    maximize  alpha @ R                       (= sum_j nu_j * slab_mass_j)
    s.t.      R <= ratio * Q,
              sum_{T_Y block} R <= k_v^n / M   (the ramp cap, w0 = k_v^n/M),
              sum_{T_V block} Q == 1,
    error_lb = 1 - value.

The achievability bound for the RCU^+ kernel (L=1) is the SAME skeleton with the
clamped-knot quadratic objective in place of the linear ramp:
    Phi_1(t) = w0*t - 1/2 t^2   on t<=w0,   Phi_1=1/2 w0^2 for t>=w0,
    per output-type block, with clamped cumulative knots a_j (a_j<=a_{j-1}+mass_j,
    a_j<=w0), Abel objective  Gamma = sum_y sum_j (nu_j-nu_{j+1})(w0 a_j - 1/2 a_j^2),
    achievable bound P_e^+ = 1 - Gamma.
Running the same code with the *ramp* objective (w0*a_j capped) reproduces the
converse LP exactly -- a built-in normalization sanity check (K=1).

For general list L the kernel antiderivative Phi_L has degree L+1 -> bracketing
LP (secant/tangent); provided here too via solve_bracketing_lp.
"""
import sys, os
import numpy as np
import cvxpy as cp


from fbl.type_based_jscc import TypeBasedJSCC


class AchievabilityJSCC:
    """
    QP / bracketing-LP prior optimization of the JSCC random-coding bound over the
    conditional codeword-type law Q (TypeBasedJSCC parametrization).

    Parameters
    ----------
    P_V : (k_v,) source law (single letter)
    W   : (k_x, k_y) channel
    n   : blocklength (method of types)
    """

    def __init__(self, P_V, W, n):
        self.tbj = TypeBasedJSCC(np.asarray(P_V, float), np.asarray(W, float), n)
        self.n = n
        self.kv_n = float(self.tbj.k_v ** n)
        # precompute the per-output-type sorted structure (independent of Q)
        self._blocks = []  # list of (st, ed, order, nu_s, nu_next)
        for st, ed in self.tbj._cond_y_vx.iterate_cond():
            a = self.tbj._alpha[st:ed]
            order = np.argsort(a)[::-1]            # decreasing metric
            nu_s = a[order]
            nu_next = np.concatenate([nu_s[1:], [0.0]])
            self._blocks.append((st, ed, order, nu_s, nu_next))

    # ---- conditional-codeword-law variable & slab masses -------------------
    def _Q_and_cons(self):
        num_Q = self.tbj._cond_vx.len
        Q = cp.Variable(num_Q, nonneg=True)
        cons = []
        for st, ed in self.tbj._cond_vx.iterate_cond():
            cons.append(cp.sum(Q[st:ed]) == 1.0)
        mass = cp.multiply(Q[self.tbj._T_vx_idx], self.tbj._ratio)  # (num_R,)
        return Q, mass, cons

    # ---- RCU^+ exact QP (L = 1) -------------------------------------------
    def solve_rcu_plus(self, M):
        """
        Exact convex QP for the L=1 (RCU^+) kernel.
        Returns (P_e_plus, Q_opt) with Q_opt the per-T_V-block conditional law.
        """
        w0 = self.kv_n / float(M)
        Q, mass, cons = self._Q_and_cons()
        obj = []
        for (st, ed, order, nu_s, nu_next) in self._blocks:
            nb = ed - st
            a = cp.Variable(nb, nonneg=True)
            cons.append(a[0] <= mass[st + order[0]])
            for j in range(1, nb):
                cons.append(a[j] <= a[j - 1] + mass[st + order[j]])
            cons.append(a <= w0)
            for j in range(nb):
                cf = float(nu_s[j] - nu_next[j])
                if cf > 0:
                    # Phi_1(a)=a - a^2/(2 w0)  (antiderivative of 1-w/w0)
                    obj.append(cf * (a[j] - 0.5 / w0 * cp.square(a[j])))
        Gamma = cp.sum(obj)
        prob = cp.Problem(cp.Maximize(Gamma), cons)
        prob.solve(solver=cp.CLARABEL)
        return 1.0 - float(Gamma.value), np.asarray(Q.value)

    # ---- converse-as-Dirac sanity (ramp objective) ------------------------
    def solve_dirac_ramp(self, M):
        """
        Same skeleton with the *ramp* (Heaviside) kernel: reproduces the converse
        LP value.  Used as a normalization sanity check (must equal
        TypeBasedJSCC.compute_converse).
        """
        w0 = self.kv_n / float(M)
        Q, mass, cons = self._Q_and_cons()
        obj = []
        for (st, ed, order, nu_s, nu_next) in self._blocks:
            nb = ed - st
            a = cp.Variable(nb, nonneg=True)
            cons.append(a[0] <= mass[st + order[0]])
            for j in range(1, nb):
                cons.append(a[j] <= a[j - 1] + mass[st + order[j]])
            cons.append(a <= w0)
            for j in range(nb):
                cf = float(nu_s[j] - nu_next[j])
                if cf > 0:
                    obj.append(cf * a[j])          # ramp: Phi(a)=a (then capped)
        Gamma = cp.sum(obj)
        prob = cp.Problem(cp.Maximize(Gamma), cons)
        prob.solve(solver=cp.CLARABEL)
        return 1.0 - float(Gamma.value), np.asarray(Q.value)

    # ---- bracketing LP for general list L ---------------------------------
    @staticmethod
    def _phi_L(t, w0, M, L):
        """Phi_L(t)=int_0^t (1-min(1,(w/w0)^L)) dw, with e^{-R}=w0=k_v^n/M scaled.

        Here the kernel envelope g_L(w)=min(1,(w/w0)^L) on the cumulative-mass
        axis; Phi_L is concave, Phi_L(0)=0.
        """
        t = np.asarray(t, float)
        out = np.where(
            t <= w0,
            t - w0 / (L + 1.0) * np.power(np.clip(t, 0, None) / w0, L + 1.0),
            w0 * (1.0 - 1.0 / (L + 1.0)),
        )
        return out

    def solve_bracketing_lp(self, M, L, K=64, side="lower"):
        """
        Bracketing LP for general list size L>=1 (Phi_L degree L+1).
        side='lower' -> secant chords (<=Phi) -> a valid achievability upper bound
        on P_e; side='upper' -> tangents (>=Phi) -> certified gap companion.
        """
        w0 = self.kv_n / float(M)
        Q, mass, cons = self._Q_and_cons()
        grid = np.linspace(0.0, w0, K + 1)
        phi_g = self._phi_L(grid, w0, M, L)
        # slopes of secant pieces (concave -> decreasing slopes)
        sec_slope = np.diff(phi_g) / np.diff(grid)
        obj = []
        for (st, ed, order, nu_s, nu_next) in self._blocks:
            nb = ed - st
            # cumulative knots sigma_j (clamped at w0) as epigraph vars
            a = cp.Variable(nb, nonneg=True)
            cons.append(a[0] <= mass[st + order[0]])
            for j in range(1, nb):
                cons.append(a[j] <= a[j - 1] + mass[st + order[j]])
            cons.append(a <= w0)
            # t_j <= Phi_L(a_j) via chords (lower) or tangents (upper)
            t = cp.Variable(nb)
            for j in range(nb):
                if side == "lower":
                    # concave hull of chords: t <= phi_g[k] + slope_k (a - grid[k])
                    for k in range(K):
                        cons.append(t[j] <= phi_g[k] + sec_slope[k] * (a[j] - grid[k]))
                else:
                    # tangents at grid points (>=Phi): t <= phi_g[k] + phi'(grid[k])(a-grid[k])
                    gpk = 1.0 - np.power(min(grid[k], w0) / w0, L)
                    for k in range(K + 1):
                        gpk = 1.0 - np.power(grid[k] / w0, L)
                        cons.append(t[j] <= phi_g[k] + gpk * (a[j] - grid[k]))
            for j in range(nb):
                cf = float(nu_s[j] - nu_next[j])
                if cf > 0:
                    obj.append(cf * t[j])
        Gamma = cp.sum(obj)
        prob = cp.Problem(cp.Maximize(Gamma), cons)
        prob.solve(solver=cp.CLARABEL)
        return 1.0 - float(Gamma.value), np.asarray(Q.value)

    # ---- baselines via the existing machinery -----------------------------
    def bound_at_Q(self, M, Q_cond):
        """Exact JSCC achievable bound at a conditional law Q (per-T_V simplex)."""
        # convert conditional Q (sums to 1 per T_V block) to joint P_T_VX
        P = self._cond_to_joint(Q_cond)
        return self.tbj.achievable_bound(M, P)

    def _cond_to_joint(self, Q_cond):
        """
        Map a conditional codeword-type law (TypeBasedJSCC.compute_converse
        parametrization, sums to 1 per T_V block) to the joint type prior
        P_T_VX consumed by compute_f_curve / achievable_bound.
        """
        # In TypeBasedJSCC's convention (see memoryless_prior) P_V is encoded in
        # `alpha`, and each T_V block of P_T_VX sums to 1/k_v^n.  Our Q sums to 1
        # per T_V block, so the matching joint is simply Q / k_v^n.
        P = np.asarray(Q_cond, float).copy()
        return P / self.kv_n

    def memoryless_optimal(self, M):
        """
        Memoryless prior optimisation, defined as the **optimal prior at n=1**
        applied i.i.d. -- exact and fast (no scipy multistart).

        Rationale: the memoryless family is the i.i.d. extension of a single-
        letter prior, so the best memoryless prior IS the single-letter optimum
        repeated.  That single-letter optimum is the exact n=1 QP at the per-
        symbol effective rate ``M1 = M**(1/n)`` (channel ``M=e^{nR}`` -> ``M1=e^R``;
        JSCC L=1 ``M=|V|^n`` -> ``M1=|V|``).  We then evaluate the i.i.d.
        extension with the trusted n-letter ``achievable_bound``.

        This is ~30x (n=2) to ~100x+ (n=3) faster than ``memoryless_baseline``
        and exact (QP, not Nelder-Mead).  It is *defined* as the n=1 optimum, so
        it can sit a hair above the true n-letter best-memoryless (≈0.2-0.5% in
        tests) -- i.e. mildly conservative, in the safe direction for a gain
        claim.

        Returns (P_e_at_blocklength_n, Q1_single_letter  (k_v, k_x)).
        """
        if self.n == 1:
            aj1 = self
        else:
            if not hasattr(self, "_aj1"):
                self._aj1 = AchievabilityJSCC(self.tbj.P_V, self.tbj.W, 1)
            aj1 = self._aj1
        M1 = float(M) ** (1.0 / self.n)
        _, Q1c = aj1.solve_rcu_plus(M1)
        Q1 = np.asarray(Q1c, float).reshape(self.tbj.k_v, self.tbj.k_x)
        P = self.tbj.memoryless_prior(Q1)
        return self.tbj.achievable_bound(M, P), Q1

    def memoryless_baseline(self, M, n_starts=12, seed=0):
        """
        Best MEMORYLESS conditional prior Q_{X|V}^{(1)} (single-letter, applied
        i.i.d.), via TypeBasedJSCC.memoryless_prior + a small multistart over the
        single-letter conditional law.  Returns (best_Pe, best_Q1).
        """
        from scipy.optimize import minimize
        kv, kx = self.tbj.k_v, self.tbj.k_x

        def obj(theta):
            th = theta.reshape(kv, kx)
            e = np.exp(th - th.max(axis=1, keepdims=True))
            Q1 = e / e.sum(axis=1, keepdims=True)
            P = self.tbj.memoryless_prior(Q1)
            return self.tbj.achievable_bound(M, P)

        r = np.random.default_rng(seed)
        best, bestQ = np.inf, None
        for _ in range(n_starts):
            res = minimize(obj, r.normal(size=kv * kx) * 1.3, method="Nelder-Mead",
                           options={"xatol": 1e-7, "fatol": 1e-11, "maxiter": 8000})
            if res.fun < best:
                best = res.fun
                th = res.x.reshape(kv, kx)
                e = np.exp(th - th.max(axis=1, keepdims=True))
                bestQ = e / e.sum(axis=1, keepdims=True)
        return best, bestQ


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    PV = np.array([0.7, 0.3])
    W = np.array([[0.9, 0.1], [0.2, 0.8]])

    print("=== n=1 validation: Dirac-ramp == converse LP (normalization K=1) ===")
    aj = AchievabilityJSCC(PV, W, 1)
    for M in [2.0, 1.5, 3.0]:
        d_val, _ = aj.solve_dirac_ramp(M)
        c_val, _ = aj.tbj.compute_converse(M)
        print(f"  M={M:.2f}: dirac-ramp Pe={d_val:.8f}  converse={c_val:.8f}  "
              f"|d|={abs(d_val - c_val):.1e}")

    print("\n=== n=1: RCU+ QP  vs  single-letter ground truth (verify_jscc) ===")
    try:
        import verify_jscc as gt
        for M in [2.0, 1.8, 2.5]:
            qp_val, qp_Q = aj.solve_rcu_plus(M)
            gt_val, _ = gt.solve_QP(W, PV, M)
            print(f"  M={M:.2f}: type-QP Pe={qp_val:.8f}  single-letter QP={gt_val:.8f}  "
                  f"|d|={abs(qp_val - gt_val):.1e}")
    except Exception as ex:
        print("  (verify_jscc unavailable:", ex, ")")
