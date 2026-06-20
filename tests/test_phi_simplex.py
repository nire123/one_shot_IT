"""
Achievability prior optimisation by the simplex march (phi_simplex), validated
against the engine's exact convex solvers.

  * channel RCU+:  the march optimum equals the exact QP optimum
                   (P_e = 1 - J, with codebook size M = e^R + 1);
  * RD exact:      the march optimum lies inside the bracketing-LP interval
                   (and is exact -- no O(1/K^2) gap);
  * the analytic water-fill gradient matches a finite difference on the simplex;
  * the optimum is no worse than the uniform start.
"""
import numpy as np
import pytest

from fbl.channel_achievable_utils import z_channel
from fbl.prioropt import phi_simplex as ps
from fbl.prioropt.achievability_qp import AchievabilityQP
from fbl.prioropt.achievability_lp_rd import AchievabilityLP_RD

LN2 = np.log(2.0)
W = z_channel(0.1)
P_X = np.array([0.75, 0.25])
D1 = np.array([[0.0, 1.0], [1.0, 0.0]])


# --------------------------------------------- channel: march == exact QP -----
@pytest.mark.parametrize("Rb", [0.15, 0.3])
def test_channel_simplex_matches_qp(Rb):
    n = 5
    R = n * Rb * LN2
    M_phi = np.exp(R) + 1.0                       # codebook size (e^R competitors)
    prog = ps.build_program("channel", W=W, n=n, kernel="rcu_plus")
    res = ps.optimize(prog, M_phi, method="pgd", max_iter=3000, tol=1e-10)
    Pe_march = 1.0 - res["J"]
    Pe_qp = AchievabilityQP(W, n).solve_rcu_plus(R)["P_e_exact"]
    assert abs(Pe_march - Pe_qp) <= 1e-6 + 1e-4 * abs(Pe_qp), (
        f"march={Pe_march:.8e} qp={Pe_qp:.8e}")


# ------------------------------------- RD: march inside bracketing-LP interval -
@pytest.mark.parametrize("Rb", [0.35, 0.55])
def test_rd_simplex_inside_bracket(Rb):
    n = 4
    M = float(np.exp(n * Rb * LN2))
    prog = ps.build_program("rd", P_X=P_X, d=D1, n=n, kernel="exact")
    res = ps.optimize(prog, M, method="pgd", max_iter=3000, tol=1e-10)
    D_march = res["J"] / n
    br = AchievabilityLP_RD(P_X, D1, n).solve_bracketing_lp(M, K=32)
    lo, hi = br["D_lo"] / n, br["D_hi"] / n
    assert lo - 1e-6 <= D_march <= hi + 1e-6, (
        f"march={D_march:.8f} not in [{lo:.8f}, {hi:.8f}]")


# ---------------------------------- analytic gradient == finite difference -----
@pytest.mark.parametrize("setting", ["channel", "rd"])
def test_gradient_matches_finite_difference(setting):
    n = 4
    if setting == "channel":
        prog = ps.build_program("channel", W=W, n=n, kernel="rcu_plus")
        M = np.exp(n * 0.3 * LN2) + 1.0
    else:
        prog = ps.build_program("rd", P_X=P_X, d=D1, n=n, kernel="exact")
        M = float(np.exp(n * 0.45 * LN2))
    rng = np.random.default_rng(0)
    Q = rng.dirichlet(np.ones(prog["num_q"]))
    _, g = ps.objective_grad(prog, Q, M)
    mu = rng.normal(size=prog["num_q"]); mu -= mu.mean()      # tangent: sum mu = 0
    eps = 1e-6
    jp = ps.objective_grad(prog, Q + eps * mu, M)[0]
    jm = ps.objective_grad(prog, Q - eps * mu, M)[0]
    fd = (jp - jm) / (2 * eps)
    assert abs(g @ mu - fd) <= 1e-4 * (1 + abs(fd)), f"analytic={g@mu:.6e} fd={fd:.6e}"


# ----------------------------------------- optimum is no worse than uniform ----
def test_optimum_beats_uniform_channel():
    n = 5
    R = n * 0.3 * LN2
    M_phi = np.exp(R) + 1.0
    prog = ps.build_program("channel", W=W, n=n, kernel="rcu_plus")
    J_unif = ps.objective_grad(prog, np.ones(prog["num_q"]) / prog["num_q"], M_phi)[0]
    res = ps.optimize(prog, M_phi, method="pgd", max_iter=3000, tol=1e-10)
    assert res["J"] >= J_unif - 1e-12          # maximising success J
