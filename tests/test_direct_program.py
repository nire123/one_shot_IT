"""
Direct simplex prior-optimization (the unified Phi-view) cross-checks.

  * directional-derivative identity on the simplex:
        (Gamma(Q+eps*mu) - Gamma(Q))/eps  ->  <grad Gamma, mu>   (sum mu = 0)
    -- exact, for both the achievability (rcu) and converse (ramp) kernels.
  * achievability direct solve agrees with the exact QP (first-order, moderate tol).
  * converse via the ramp kernel is 0 at a fully-achievable low rate.
"""
import numpy as np
import pytest

from fbl.channel_achievable_utils import z_channel
from fbl.prioropt import AchievabilityQP, DirectPriorOpt

W = z_channel(0.1)


@pytest.mark.parametrize("kernel", ["rcu", "converse"])
def test_directional_derivative_identity(kernel):
    n = 6
    dp = DirectPriorOpt(W, n)
    R = n * 0.3 * np.log(2.0)
    w0 = float(np.exp(-R))
    rng = np.random.default_rng(0)
    Q = rng.random(dp.num_q); Q /= Q.sum()
    mu = rng.standard_normal(dp.num_q); mu -= mu.mean()      # sum(mu) = 0
    G0, g = dp._gamma_grad(Q, w0, kernel)
    eps = 1e-7
    fd = (dp._gamma_grad(Q + eps * mu, w0, kernel)[0] - G0) / eps
    assert abs(fd - g @ mu) < 1e-4
    # only the centered gradient acts (sum mu = 0)
    assert abs(g @ mu - (g - g.mean()) @ mu) < 1e-9


def test_achievability_direct_matches_qp():
    n = 6
    dp = DirectPriorOpt(W, n)
    aqp = AchievabilityQP(W, n)
    for Rb in (0.1, 0.4):
        R = n * Rb * np.log(2.0)
        pe = dp.solve(R, kernel="rcu", method="pgd", max_iter=3000, tol=1e-9)["P_e"]
        qp = aqp.solve_rcu_plus(R)["P_e_exact"]
        assert qp - 1e-6 <= pe <= qp + 5e-3        # direct >= optimum, converges to it


def test_converse_zero_at_low_rate():
    n = 6
    dp = DirectPriorOpt(W, n)
    R = n * 0.05 * np.log(2.0)                      # low rate: converse is fully achievable
    res = dp.solve(R, kernel="converse", method="pgd", max_iter=3000, tol=1e-10)
    assert res["P_e"] < 1e-5
