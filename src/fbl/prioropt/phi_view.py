r"""
The unified Phi-view of the PEP spectrum:  J = c^T Phi(A Q).

This module is the *preprocessing stage* for the thesis nonlinear program
(chapters ``p1var`` / ``p1nlp``): the value-weighted spectrum, and every kernel
average of it, is one program in the prior,

        J  =  c^T Phi(A Q),

where ``A Q`` is the cumulative-rank profile (linear in the prior ``Q``), ``c``
the non-negative value-gaps, and the **kernel chooses the fixed potential Phi**
(``Phi' = int_sigma^1 kappa``).  Two tails occur:

    * error tail   (channel/JSCC): profile = survival  sigma(nu)=Q{m>=nu},
      Phi concave, the bound MAXIMISES J;
    * correct tail (rate-distortion): profile = CDF  tau(nu)=Q{d<=nu},
      Phi convex, the bound MINIMISES J.

``preprocess_channel`` / ``preprocess_rd`` build the literal ``(A, c)`` (with the
metric order fixed, so the prior enters only through ``A Q``); the registries map
each bound to its ``Phi`` (and, for the error tail, its kernel ``kappa``).  The
identity is verified numerically in ``tests/test_phi_view.py``: ``J_formula`` (the
closed form ``c^T Phi(A Q)``) equals ``J_direct`` (an independent quadrature of the
operational bound).

Convention: ``M`` is the codebook size (``M-1`` competitors); the RCU+ kernel is
uniform on ``w <= 1/(M-1)``.
"""
import numpy as np

# ------------------------------------------------------------------ potentials
# Error tail (channel): Phi concave, argument sigma = survival mass.
def phi_channel_exact(sigma, M):
    """Exact random-coding potential, Phi(sigma) = (1-(1-sigma)^M)/M."""
    return (1.0 - np.maximum(1.0 - sigma, 0.0) ** M) / M


def kappa_channel_exact(w, M):
    """Kernel of the exact bound: kappa(w) = (M-1)(1-w)^{M-2} (so int_0^1 = 1)."""
    return (M - 1.0) * np.maximum(1.0 - w, 0.0) ** (M - 2.0)


def phi_rcu_plus(sigma, M):
    """RCU+ (upper bound) potential, Phi(sigma)=sigma-(M-1)/2 sigma^2, clamped."""
    w0 = 1.0 / (M - 1.0)
    return np.where(sigma <= w0, sigma - 0.5 * (M - 1.0) * sigma ** 2, 0.5 * w0)


def kappa_rcu_plus(w, M):
    """Kernel of RCU+: uniform height (M-1) on w <= 1/(M-1), else 0."""
    return np.where(w <= 1.0 / (M - 1.0), M - 1.0, 0.0)


# Correct tail (rate-distortion): Phi convex, argument tau = CDF mass.
def phi_rd_exact(tau, M):
    """Exact best-of-M potential, Phi(tau) = (1-tau)^M."""
    return np.maximum(1.0 - tau, 0.0) ** M


def phi_rd_smooth(tau, M):
    """Smooth upper bound, Phi(tau) = e^{-M tau} >= (1-tau)^M."""
    return np.exp(-M * tau)


CHANNEL_KERNELS = {                       # name -> (Phi, kappa)
    "exact":    (phi_channel_exact, kappa_channel_exact),
    "rcu_plus": (phi_rcu_plus,      kappa_rcu_plus),
}
RD_KERNELS = {                            # name -> Phi  (direct = operational)
    "exact":  phi_rd_exact,
    "smooth": phi_rd_smooth,
}


# ----------------------------------------------------------------- preprocess
def preprocess_channel(W, Q_X):
    r"""
    Build the Phi-view data for channel coding (error tail).

    For each output ``y`` the inputs are ordered by decreasing metric
    ``m(x,y)=W(y|x)``; the cumulative-rank rows (0/1) and the value-gaps
    ``c_j=m_j-m_{j+1}`` are stacked over ``y``.  The metric order fixes ``A`` and
    ``c``; the prior enters only through ``sigma = A Q``.

    Returns dict(A, c, blocks, Q) where ``blocks`` is a list of per-output
    ``(values_sorted, masses_sorted, sigma_cumulative)`` used by the direct path.
    """
    W = np.asarray(W, float)
    Q = np.asarray(Q_X, float)
    kx, ky = W.shape
    A_rows, c_list, blocks = [], [], []
    for y in range(ky):
        m = W[:, y]
        order = np.argsort(m)[::-1]               # decreasing metric
        mv, qv = m[order], Q[order]
        sig = np.cumsum(qv)
        cum = np.zeros(kx)
        for j in range(kx):
            cum[order[j]] = 1.0                    # top-j set indicator
            A_rows.append(cum.copy())
            m_next = mv[j + 1] if j + 1 < kx else 0.0
            c_list.append(mv[j] - m_next)          # value-gap >= 0
        blocks.append((mv, qv, sig))
    return {"A": np.array(A_rows), "c": np.array(c_list), "blocks": blocks, "Q": Q}


def preprocess_rd(P_X, d, Q_Y):
    r"""
    Build the Phi-view data for rate-distortion (correct tail).

    For each source ``x`` the reproductions are ordered by increasing distortion
    ``d(x,y)``; the CDF rows (0/1) and the distortion-gaps weighted by ``P_X(x)``
    are stacked.  An offset row (sigma=0, Phi=1) carries the minimum distortion.
    The prior enters only through ``tau = A Q``.

    Returns dict(A, c, blocks, Q) where ``blocks`` is a list of per-source
    ``(distortions_sorted, masses_sorted, weight=P_X(x))``.
    """
    P_X = np.asarray(P_X, float)
    d = np.asarray(d, float)
    Q = np.asarray(Q_Y, float)
    kx, ky = d.shape
    A_rows, c_list, blocks = [], [], []
    for x in range(kx):
        order = np.argsort(d[x, :])                # increasing distortion
        dv, qv = d[x, order], Q[order]
        # offset: interval [0, dv[0]) has tau=0 (Phi(0)=1) -> pays dv[0]
        A_rows.append(np.zeros(ky))
        c_list.append(P_X[x] * dv[0])
        cum = np.zeros(ky)
        for j in range(ky - 1):
            cum[order[j]] = 1.0                    # CDF up to sorted index j
            A_rows.append(cum.copy())              # tau_j = Q{d <= dv[j]}
            c_list.append(P_X[x] * (dv[j + 1] - dv[j]))
        blocks.append((dv, qv, float(P_X[x])))
    return {"A": np.array(A_rows), "c": np.array(c_list), "blocks": blocks, "Q": Q}


# -------------------------------------------------------------- the two values
def J_formula(pre, M, kernel, setting):
    """Closed form  J = c^T Phi(A Q)."""
    phi = (CHANNEL_KERNELS[kernel][0] if setting == "channel"
           else RD_KERNELS[kernel])
    sigma = pre["A"] @ pre["Q"]
    return float(pre["c"] @ phi(sigma, M))


def J_direct_channel(pre, M, kernel, ngrid=100001):
    """Independent quadrature: int_0^1 F(w) kappa(w) dw, F the raw spectrum."""
    _, kappa = CHANNEL_KERNELS[kernel]
    w = np.linspace(0.0, 1.0, ngrid)
    F = np.zeros_like(w)
    for mv, qv, sig in pre["blocks"]:
        s_prev = 0.0
        for j in range(len(mv)):
            F += mv[j] * np.clip(w - s_prev, 0.0, qv[j])   # mass of cand j <= w
            s_prev = sig[j]
    return float(np.trapz(F * kappa(w, M), w))


def J_direct_rd(pre, M, kernel, ngrid=100001):
    """Independent quadrature: sum_x P_X(x) int_0^inf Phi(tau_x(t)) dt (best-of-M)."""
    D = 0.0
    for dv, qv, wt in pre["blocks"]:
        tmax = float(dv[-1])
        if tmax <= 0.0:
            continue
        t = np.linspace(0.0, tmax, ngrid)
        idx = np.searchsorted(dv, t, side="right")          # #{d <= t}
        cum_q = np.concatenate([[0.0], np.cumsum(qv)])
        tau = np.minimum(cum_q[idx], 1.0)
        integ = (np.maximum(1.0 - tau, 0.0) ** M if kernel == "exact"
                 else np.exp(-M * tau))
        D += wt * np.trapz(integ, t)
    return float(D)


def J_direct(pre, M, kernel, setting, ngrid=100001):
    """Dispatch the independent quadrature by setting."""
    if setting == "channel":
        return J_direct_channel(pre, M, kernel, ngrid)
    return J_direct_rd(pre, M, kernel, ngrid)


# ----------------------------------------------------- type-based (method of types)
# The same identity J = c^T Phi(A Q) holds on the type-based staircase, where the
# "candidates" are (output-type, conditional-type) slabs: the metric order fixes
# the value-gaps c and the slab masses ratio*Q give the cumulative profile sigma.
# These evaluators read the staircase from the existing engines and apply the same
# potentials, so a memoryless type prior reproduces the lifted one-shot value
# (verified to machine precision in tests/test_phi_view.py).
def J_typebased_channel(W, n, Q_type, M, kernel):
    """c^T Phi(A Q) on the type-based channel staircase (error tail)."""
    from fbl.prioropt.achievability_qp import AchievabilityQP
    phi = CHANNEL_KERNELS[kernel][0]
    Q = np.asarray(Q_type, float)
    J = 0.0
    for nu, ridx, ratio in AchievabilityQP(W, n)._blocks():
        sigma = np.cumsum(ratio * Q[ridx])              # cumulative profile
        c = nu - np.append(nu[1:], 0.0)                 # value-gaps >= 0
        J += float(np.sum(c * phi(sigma, M)))
    return J


def J_typebased_rd(P_X, d, n, Q_type, M, kernel):
    """c^T Phi(A Q) on the type-based RD staircase (correct tail)."""
    from fbl.prioropt.achievability_lp_rd import AchievabilityLP_RD
    phi = RD_KERNELS[kernel]
    Q = np.asarray(Q_type, float)
    J = 0.0
    for delta, ridx, ratio in AchievabilityLP_RD(P_X, d, n)._blocks():
        tau = np.cumsum(ratio * Q[ridx])                # coverage (ascending dist.)
        J += float(delta[0])                            # offset (Phi(0)=1)
        J += float(np.sum(np.diff(delta) * phi(tau[:-1], M)))
    return J
