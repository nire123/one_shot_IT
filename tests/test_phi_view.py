"""
Validate the foundational Phi-view identity  J = c^T Phi(A Q).

For each case -- channel coding / rate-distortion, exact / upper-bound kernel --
fix a prior and check that the closed form ``J_formula = c^T Phi(A Q)`` equals an
*independent* quadrature ``J_direct`` of the operational bound:
  * channel: int_0^1 F(w) kappa(w) dw  (raw spectrum against the kernel);
  * RD:      sum_x P_X(x) int_0^inf Phi(tau_x(t)) dt  (operational best-of-M).

This is the numerical proof of Lemma p1var/p1nlp before any optimisation is built
on top of it.
"""
import numpy as np
import pytest

from fbl.prioropt.phi_view import (
    preprocess_channel, preprocess_rd, J_formula, J_direct,
)

# ----------------------------------------------------------------- fixtures ---
CHANNELS = {
    "BSC(0.1)":   np.array([[0.9, 0.1], [0.1, 0.9]]),
    "Z(0.1)":     np.array([[0.9, 0.1], [0.0, 1.0]]),
    "rand3x3":    np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]]),
    "asym2x3":    np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]]),
}
CH_PRIORS = {2: [np.array([0.5, 0.5]), np.array([0.7, 0.3])],
             3: [np.ones(3) / 3, np.array([0.5, 0.3, 0.2])]}

# binary source + Hamming (2x2), and a 3-symbol distortion
RD_PROBLEMS = {
    "bms_hamming": (np.array([0.75, 0.25]), np.array([[0.0, 1.0], [1.0, 0.0]])),
    "rd3x3": (np.array([0.5, 0.3, 0.2]),
              np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])),
}
RD_PRIORS = {2: [np.array([0.5, 0.5]), np.array([0.6, 0.4])],
             3: [np.ones(3) / 3, np.array([0.2, 0.3, 0.5])]}

CH_M = [2.0, 3.0, 5.0, 8.0]
RD_M = [2.0, 4.0, 8.0]
TOL = 2e-4               # quadrature error of J_direct on a 1e5 grid


# ----------------------------------------------- channel: formula == direct ---
@pytest.mark.parametrize("kernel", ["exact", "rcu_plus"])
@pytest.mark.parametrize("name", list(CHANNELS))
def test_channel_formula_matches_direct(name, kernel):
    W = CHANNELS[name]
    for Q in CH_PRIORS[W.shape[0]]:
        pre = preprocess_channel(W, Q)
        for M in CH_M:
            jf = J_formula(pre, M, kernel, "channel")
            jd = J_direct(pre, M, kernel, "channel")
            assert abs(jf - jd) <= TOL + 1e-3 * abs(jd), (
                f"{name} {kernel} M={M}: formula={jf:.6g} direct={jd:.6g}")


# ----------------------------------------------------- RD: formula == direct ---
@pytest.mark.parametrize("kernel", ["exact", "smooth"])
@pytest.mark.parametrize("name", list(RD_PROBLEMS))
def test_rd_formula_matches_direct(name, kernel):
    P_X, d = RD_PROBLEMS[name]
    for Q in RD_PRIORS[d.shape[1]]:
        pre = preprocess_rd(P_X, d, Q)
        for M in RD_M:
            jf = J_formula(pre, M, kernel, "rd")
            jd = J_direct(pre, M, kernel, "rd")
            assert abs(jf - jd) <= TOL + 1e-3 * abs(jd), (
                f"{name} {kernel} M={M}: formula={jf:.6g} direct={jd:.6g}")


# ----------------------------------------- upper-bound kernels actually bound --
def test_channel_rcu_plus_lower_bounds_success():
    """RCU+ is an error upper bound, hence J_rcu+ <= J_exact (less success)."""
    W = CHANNELS["Z(0.1)"]
    pre = preprocess_channel(W, np.array([0.6, 0.4]))
    for M in CH_M:
        assert (J_formula(pre, M, "rcu_plus", "channel")
                <= J_formula(pre, M, "exact", "channel") + 1e-9)


def test_rd_smooth_upper_bounds_distortion():
    """Smooth bound uses (1-tau)^M <= e^{-M tau}, hence D_smooth >= D_exact."""
    P_X, d = RD_PROBLEMS["bms_hamming"]
    pre = preprocess_rd(P_X, d, np.array([0.5, 0.5]))
    for M in RD_M:
        assert (J_formula(pre, M, "smooth", "rd")
                >= J_formula(pre, M, "exact", "rd") - 1e-9)


# ------------ tie RD-exact to the trusted lifted one-shot path (= Monte-Carlo) --
def test_rd_exact_matches_oneshot():
    """The exact best-of-M distortion equals the lifted one-shot engine's
    ``theory`` (itself validated against Monte-Carlo). At n=1 the lifted alphabet
    is the reproduction alphabet, so phi_view(exact) must match it."""
    from fbl import OneShotRD
    for name in RD_PROBLEMS:
        P_X, d = RD_PROBLEMS[name]
        osr = OneShotRD(P_X, d)                       # n=1 lifted == single-letter
        for Q in RD_PRIORS[d.shape[1]]:
            curve = osr.compute_curve(Q)
            pre = preprocess_rd(P_X, d, Q)
            for M in RD_M:
                jf = J_formula(pre, M, "exact", "rd")
                ref = osr.theory(curve, M)               # numerical integrator
                assert abs(jf - ref) <= 1e-5 + 1e-3 * abs(ref), (
                    f"{name} M={M}: phi_view={jf:.6g} one-shot={ref:.6g}")
