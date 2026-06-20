r"""
Achievability prior optimisation by a first-order march on the simplex.

Built on the validated Phi-view relaxation (``phi_view``): the bound is
``J(Q) = c^T Phi(A Q)`` over the probability simplex, concave for the channel
error tail (maximise) and convex for the rate-distortion correct tail (minimise).
Both are convex programs; we solve them *directly on the simplex* with projected
gradient (or Frank-Wolfe), using the analytic water-fill gradient

        grad J(x) = sum_blocks sum_{i fed by x} ratio_i sum_{j>=i} c_j Phi'(sigma_j),

i.e. ``A^T (c .* Phi'(A Q))`` evaluated block by block (no dense ``A``).  This is
the general, scalable, exact-for-any-kernel realisation of the unified program --
the same first-order method works for every kernel because only ``Phi`` changes.

We optimise on the **type-based** staircase (the method-of-types representation),
so the returned ``Q`` is a type prior directly comparable to the engine's QP /
bracketing-LP optima.  ``sense=+1`` maximises ``J`` (channel success), ``sense=-1``
minimises ``J`` (RD distortion); internally we always ascend ``sense*J``.
"""
import numpy as np

from fbl.prioropt import phi_view as pv


def _project_simplex(v):
    """Euclidean projection of v onto {x >= 0, sum x = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - css / ind > 0
    rho = ind[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def build_program(setting, *, W=None, P_X=None, d=None, n=None, kernel="exact"):
    """
    Assemble the simplex program from the type-based staircase.

    Returns dict(blocks, num_q, phi, dphi, sense) where each block is
    ``(c, ridx, ratio, offset)`` so that ``J = sum_b offset_b + c_b . Phi(sigma_b)``
    with ``sigma_b = cumsum(ratio_b * Q[ridx_b])``.
    """
    if setting == "channel":
        from fbl.prioropt.achievability_qp import AchievabilityQP
        aqp = AchievabilityQP(W, n)
        phi, dphi = pv.CHANNEL_KERNELS[kernel][0], pv.CHANNEL_DERIV[kernel]
        blocks = []
        for nu, ridx, ratio in aqp._blocks():
            c = nu - np.append(nu[1:], 0.0)               # value-gaps >= 0
            blocks.append((c, ridx, ratio, 0.0))
        return {"blocks": blocks, "num_q": aqp.tb.num_q,
                "phi": phi, "dphi": dphi, "sense": +1}     # maximise success
    elif setting == "rd":
        from fbl.prioropt.achievability_lp_rd import AchievabilityLP_RD
        alr = AchievabilityLP_RD(P_X, d, n)
        phi, dphi = pv.RD_KERNELS[kernel], pv.RD_DERIV[kernel]
        blocks = []
        for delta, ridx, ratio in alr._blocks():
            c = np.append(np.diff(delta), 0.0)            # gaps; last term inert
            blocks.append((c, ridx, ratio, float(delta[0])))
        return {"blocks": blocks, "num_q": alr.tb.num_q,
                "phi": phi, "dphi": dphi, "sense": -1}     # minimise distortion
    raise ValueError(setting)


def objective_grad(prog, Q, M):
    """Return (J, grad J) at type prior Q via the block water-fill formula."""
    J = 0.0
    g = np.zeros(prog["num_q"])
    phi, dphi = prog["phi"], prog["dphi"]
    for c, ridx, ratio, offset in prog["blocks"]:
        sigma = np.cumsum(ratio * Q[ridx])
        J += offset + float(np.sum(c * phi(sigma, M)))
        suffix = np.cumsum((c * dphi(sigma, M))[::-1])[::-1]   # sum_{j>=i}
        np.add.at(g, ridx, ratio * suffix)
    return J, g


def check_kkt(prog, Q, M, support_tol=1e-6, tol=1e-5):
    r"""
    Certify Phi-view optimality of a candidate prior Q by the water-filling KKT
    condition, independently of any other solver.

    Maximising ``f = sense * J`` (concave) over the simplex, Q is optimal iff the
    gradient ``grad f`` is flat on the support and dominated off it::

        grad f_x = lambda      for x with Q_x > 0     (stationary)
        grad f_x <= lambda     for x with Q_x = 0     (dual feasible)
        lambda = max_x grad f_x

    Returns dict(kkt, stationary, dual_feasible, support_spread,
    off_support_excess, fw_gap, lambda, support_size).
    """
    Q = np.asarray(Q, float)
    _, g = objective_grad(prog, Q, M)
    gf = prog["sense"] * g                              # ascent gradient of f
    lam = float(gf.max())
    supp = Q > support_tol
    spread = float(gf[supp].max() - gf[supp].min()) if supp.any() else 0.0
    off_excess = float((gf[~supp] - lam).max()) if (~supp).any() else -np.inf
    fw_gap = float(lam - gf @ Q)
    stationary = spread <= tol
    dual_feasible = off_excess <= tol
    return {"kkt": bool(stationary and dual_feasible),
            "stationary": bool(stationary), "dual_feasible": bool(dual_feasible),
            "support_spread": spread, "off_support_excess": off_excess,
            "fw_gap": fw_gap, "lambda": lam, "support_size": int(supp.sum())}


def optimize(prog, M, method="pgd", max_iter=5000, tol=1e-11,
             obj_tol=1e-13, patience=8, warm_start=None, history=False):
    """
    Maximise ``sense*J`` over the simplex by a first-order march.

    method : 'pgd' (projected gradient + backtracking) or 'fw' (Frank-Wolfe).
    Stops when the Frank-Wolfe optimality gap < ``tol`` OR the objective stalls
    (relative improvement < ``obj_tol`` for ``patience`` iterations -- the flat
    clamp region of the achievability potentials, where the gap plateaus although
    the optimum is already reached).
    Returns dict(Q, J, gap, iters, sense) (+ 'hist' of objective if history).
    """
    sense = prog["sense"]
    nq = prog["num_q"]
    Q = (np.ones(nq) / nq if warm_start is None
         else _project_simplex(np.asarray(warm_start, float)))
    eta, gap, it, hist = 1.0, np.inf, 0, []
    f_prev, stall = -np.inf, 0
    for k in range(max_iter):
        J, g = objective_grad(prog, Q, M)
        gf = sense * g                                 # ascent on f = sense*J
        if history:
            hist.append(J)
        x = int(np.argmax(gf))
        gap = float(gf[x] - gf @ Q)                    # Frank-Wolfe optimality gap
        it = k + 1
        f = sense * J
        if f - f_prev <= obj_tol * (1.0 + abs(f)):     # objective plateau
            stall += 1
        else:
            stall = 0
        f_prev = f
        if gap < tol or stall >= patience:
            break
        if method == "fw":
            dvec = -Q.copy(); dvec[x] += 1.0           # toward vertex x
            a, b, gr = 0.0, 1.0, 0.6180339887
            cc, ee = b - gr * (b - a), a + gr * (b - a)
            fc = sense * objective_grad(prog, Q + cc * dvec, M)[0]
            fe = sense * objective_grad(prog, Q + ee * dvec, M)[0]
            for _ in range(40):
                if fc < fe:
                    a, cc, fc = cc, ee, fe
                    ee = a + gr * (b - a)
                    fe = sense * objective_grad(prog, Q + ee * dvec, M)[0]
                else:
                    b, ee, fe = ee, cc, fc
                    cc = b - gr * (b - a)
                    fc = sense * objective_grad(prog, Q + cc * dvec, M)[0]
            Q = Q + 0.5 * (a + b) * dvec
        else:                                          # projected gradient
            f = sense * J
            step = eta
            for _ in range(60):
                Qn = _project_simplex(Q + step * gf)
                fn = sense * objective_grad(prog, Qn, M)[0]
                if fn >= f + 1e-4 * step * (gf @ (Qn - Q)):
                    break
                step *= 0.5
            Q = Qn
            eta = min(step * 2.0, 1e6)
    J, _ = objective_grad(prog, Q, M)
    out = {"Q": Q, "J": J, "gap": gap, "iters": it, "sense": sense,
           "kkt": check_kkt(prog, Q, M)}
    if history:
        out["hist"] = hist
    return out


# convenience one-call wrappers ------------------------------------------------
def optimize_channel(W, n, M, kernel="rcu_plus", **kw):
    prog = build_program("channel", W=W, n=n, kernel=kernel)
    return optimize(prog, M, **kw)


def optimize_rd(P_X, d, n, M, kernel="exact", **kw):
    prog = build_program("rd", P_X=P_X, d=d, n=n, kernel=kernel)
    return optimize(prog, M, **kw)
