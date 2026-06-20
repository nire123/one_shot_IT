r"""
Achievability prior optimisation by a first-order march on the simplex.

Built on the validated Phi-view relaxation (``phi_view``): the bound is
``J(Q) = c^T Phi(A Q)`` over a product of probability simplices, concave for the
channel/JSCC error tail (maximise) and convex for the rate-distortion correct
tail (minimise).  We solve it *directly on the simplex* with projected gradient
(or Frank-Wolfe), using the analytic water-fill gradient

        grad J(x) = sum_blocks sum_{i fed by x} ratio_i sum_{j>=i} c_j Phi'(sigma_j),

i.e. ``A^T (c .* Phi'(A Q))`` evaluated block by block (no dense ``A``).  This is
the general, scalable, exact-for-any-kernel realisation of the unified program --
the same first-order method works for every setting because only ``Phi`` and the
simplex structure change.

The prior lives on the **type-based** staircase.  ``simplex_blocks`` partitions
the prior variable into the simplices it must satisfy: a single global simplex for
channel and rate-distortion, and one per source-type block for JSCC (the
conditional codeword-type law).  ``sense=+1`` maximises ``J`` (channel/JSCC
success), ``sense=-1`` minimises ``J`` (RD distortion); internally we always
ascend ``sense*J``.
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


def _project_blocks(v, blocks):
    """Project each segment v[s:e] onto its own probability simplex."""
    out = np.array(v, float)
    for s, e in blocks:
        out[s:e] = _project_simplex(out[s:e])
    return out


def build_program(setting, *, W=None, P_X=None, d=None, n=None, P_V=None,
                  M=None, kernel="exact"):
    """
    Assemble the simplex program from the type-based staircase.

    Returns dict(blocks, num_q, phi, dphi, sense, simplex_blocks) where each
    staircase block is ``(c, ridx, ratio, offset)`` so that
    ``J = sum_b offset_b + c_b . Phi(sigma_b)``, ``sigma_b = cumsum(ratio_b * Q[ridx_b])``,
    and ``simplex_blocks`` lists the (start, end) simplices the prior obeys.

    JSCC needs ``P_V``, ``W``, ``n`` and the codebook size ``M`` (its potential
    depends on the threshold ``w0 = k_v^n / M``); the kernel is fixed to RCU+.
    """
    if setting == "channel":
        from fbl.prioropt.achievability_qp import AchievabilityQP
        aqp = AchievabilityQP(W, n)
        phi, dphi = pv.CHANNEL_KERNELS[kernel][0], pv.CHANNEL_DERIV[kernel]
        blocks = []
        for nu, ridx, ratio in aqp._blocks():
            c = nu - np.append(nu[1:], 0.0)               # value-gaps >= 0
            blocks.append((c, ridx, ratio, 0.0))
        nq = aqp.tb.num_q
        return {"blocks": blocks, "num_q": nq, "phi": phi, "dphi": dphi,
                "sense": +1, "simplex_blocks": [(0, nq)]}     # maximise success
    elif setting == "rd":
        from fbl.prioropt.achievability_lp_rd import AchievabilityLP_RD
        alr = AchievabilityLP_RD(P_X, d, n)
        phi, dphi = pv.RD_KERNELS[kernel], pv.RD_DERIV[kernel]
        blocks = []
        for delta, ridx, ratio in alr._blocks():
            c = np.append(np.diff(delta), 0.0)            # gaps; last term inert
            blocks.append((c, ridx, ratio, float(delta[0])))
        nq = alr.tb.num_q
        return {"blocks": blocks, "num_q": nq, "phi": phi, "dphi": dphi,
                "sense": -1, "simplex_blocks": [(0, nq)]}     # minimise distortion
    elif setting == "jscc":
        from fbl.prioropt.achievability_jscc import AchievabilityJSCC
        aj = AchievabilityJSCC(P_V, W, n)
        tbj = aj.tbj
        w0 = aj.kv_n / float(M)
        phi = lambda s, _M: pv.phi_jscc_rcu(s, w0)        # threshold-parametrised
        dphi = lambda s, _M: pv.dphi_jscc_rcu(s, w0)
        blocks = []
        for (st, ed, order, nu_s, nu_next) in aj._blocks:
            ridx = tbj._T_vx_idx[st:ed][order]
            ratio = tbj._ratio[st:ed][order]
            blocks.append((nu_s - nu_next, ridx, ratio, 0.0))
        nq = tbj._cond_vx.len
        simplex_blocks = list(tbj._cond_vx.iterate_cond())   # one per source type
        return {"blocks": blocks, "num_q": nq, "phi": phi, "dphi": dphi,
                "sense": +1, "simplex_blocks": simplex_blocks}
    raise ValueError(setting)


def objective_grad(prog, Q, M):
    """Return (J, grad J) at prior Q via the block water-fill formula."""
    J = 0.0
    g = np.zeros(prog["num_q"])
    phi, dphi = prog["phi"], prog["dphi"]
    for c, ridx, ratio, offset in prog["blocks"]:
        sigma = np.cumsum(ratio * Q[ridx])
        J += offset + float(np.sum(c * phi(sigma, M)))
        suffix = np.cumsum((c * dphi(sigma, M))[::-1])[::-1]   # sum_{j>=i}
        np.add.at(g, ridx, ratio * suffix)
    return J, g


def _uniform_start(prog):
    Q = np.zeros(prog["num_q"])
    for s, e in prog["simplex_blocks"]:
        Q[s:e] = 1.0 / (e - s)
    return Q


def _fw_gap(prog, gf, Q):
    """Frank-Wolfe optimality gap, summed over the product of simplices."""
    gap = 0.0
    for s, e in prog["simplex_blocks"]:
        gap += float(gf[s:e].max() - gf[s:e] @ Q[s:e])
    return gap


def check_kkt(prog, Q, M, support_tol=1e-6, tol=1e-5):
    r"""
    Certify Phi-view optimality of a candidate prior Q by the water-filling KKT
    condition, **per simplex block**, independently of any other solver.

    On each block, maximising ``f = sense * J`` is optimal iff the gradient is
    flat on the block's support and dominated off it::

        grad f_x = lambda_b   (x in block b, Q_x > 0),
        grad f_x <= lambda_b  (x in block b, Q_x = 0),   lambda_b = max over block.

    Returns dict(kkt, stationary, dual_feasible, support_spread,
    off_support_excess, fw_gap, support_size).
    """
    Q = np.asarray(Q, float)
    _, g = objective_grad(prog, Q, M)
    gf = prog["sense"] * g
    spread = 0.0
    off_excess = -np.inf
    supp_size = 0
    for s, e in prog["simplex_blocks"]:
        gb, qb = gf[s:e], Q[s:e]
        lam = float(gb.max())
        sup = qb > support_tol
        supp_size += int(sup.sum())
        if sup.any():
            spread = max(spread, float(gb[sup].max() - gb[sup].min()))
        if (~sup).any():
            off_excess = max(off_excess, float((gb[~sup] - lam).max()))
    stationary = spread <= tol
    dual_feasible = off_excess <= tol
    return {"kkt": bool(stationary and dual_feasible),
            "stationary": bool(stationary), "dual_feasible": bool(dual_feasible),
            "support_spread": spread, "off_support_excess": off_excess,
            "fw_gap": _fw_gap(prog, gf, Q), "support_size": supp_size}


def optimize(prog, M, method="pgd", max_iter=5000, tol=1e-11,
             obj_tol=1e-13, patience=8, warm_start=None, history=False):
    """
    Maximise ``sense*J`` over the product of simplices by a first-order march.

    Stops when the Frank-Wolfe gap < ``tol`` OR the objective stalls (the flat
    clamp region, where the gap plateaus though the optimum is reached).
    Returns dict(Q, J, gap, iters, sense, kkt) (+ 'hist' if history).
    """
    sense = prog["sense"]
    sb = prog["simplex_blocks"]
    Q = (_uniform_start(prog) if warm_start is None
         else _project_blocks(np.asarray(warm_start, float), sb))
    eta, gap, it, hist = 1.0, np.inf, 0, []
    f_prev, stall = -np.inf, 0
    for k in range(max_iter):
        J, g = objective_grad(prog, Q, M)
        gf = sense * g
        if history:
            hist.append(J)
        gap = _fw_gap(prog, gf, Q)
        it = k + 1
        f = sense * J
        if f - f_prev <= obj_tol * (1.0 + abs(f)):
            stall += 1
        else:
            stall = 0
        f_prev = f
        if gap < tol or stall >= patience:
            break
        if method == "fw":
            d = -Q.copy()                                  # per-block FW vertex
            for s, e in sb:
                d[s + int(np.argmax(gf[s:e]))] += 1.0
            a, b, gr = 0.0, 1.0, 0.6180339887
            cc, ee = b - gr * (b - a), a + gr * (b - a)
            fc = sense * objective_grad(prog, Q + cc * d, M)[0]
            fe = sense * objective_grad(prog, Q + ee * d, M)[0]
            for _ in range(40):
                if fc < fe:
                    a, cc, fc = cc, ee, fe
                    ee = a + gr * (b - a)
                    fe = sense * objective_grad(prog, Q + ee * d, M)[0]
                else:
                    b, ee, fe = ee, cc, fc
                    cc = b - gr * (b - a)
                    fc = sense * objective_grad(prog, Q + cc * d, M)[0]
            Q = Q + 0.5 * (a + b) * d
        else:                                              # projected gradient
            step = eta
            for _ in range(60):
                Qn = _project_blocks(Q + step * gf, sb)
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
    return optimize(build_program("channel", W=W, n=n, kernel=kernel), M, **kw)


def optimize_rd(P_X, d, n, M, kernel="exact", **kw):
    return optimize(build_program("rd", P_X=P_X, d=d, n=n, kernel=kernel), M, **kw)


def optimize_jscc(P_V, W, n, M, **kw):
    prog = build_program("jscc", P_V=P_V, W=W, n=n, M=M)
    return optimize(prog, M, **kw)
