"""
Prior-optimization cross-checks (achievability QP / bracketing LP).

Self-contained invariants (no external ground-truth files):

  * channel : exact QP <= best memoryless prior; bracketing LP straddles the
    exact QP (P_lo <= P_e <= P_hi).
  * RD      : bracketing LP straddles the exact best-of-M distortion at its own
    optimal prior (D_lo <= D_exact <= D_hi).
  * JSCC    : Dirac-kernel program == meta-converse LP; exact QP <= the
    "n=1-optimal applied i.i.d." memoryless baseline (nesting guarantee);
    converse <= achievable.
  * excess  : bracketing LP straddles the exact P_exc at its optimal prior.
"""
import numpy as np
import pytest

from fbl import TypeBasedChannel, TypeBasedJSCC
from fbl.type_based_utils import memoryless_to_type_prior
from fbl.channel_achievable_utils import z_channel, binary_symmetric_channel
from fbl.prioropt import (
    AchievabilityQP, AchievabilityLP_RD, AchievabilityJSCC, ExcessRD,
)
from fbl.prioropt.typebased_block_lp import rcu_plus_from_F_curve

TOL = 1e-6
Zc = z_channel(0.1)


# ── channel ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("Rbits", [0.1, 0.3, 0.5])
def test_channel_qp_le_best_memoryless(Rbits):
    n = 8
    W = Zc
    aqp = AchievabilityQP(W, n)
    tbc = TypeBasedChannel(W, n)
    R = n * Rbits * np.log(2)          # total rate (nats); M = e^R
    M = float(np.exp(R))
    qp = aqp.solve_rcu_plus(R)["P_e_exact"]
    # memoryless must use the SAME RCU+ kernel as the QP (w0 = 1/M)
    best_ml = min(
        rcu_plus_from_F_curve(*tbc.compute_curve(memoryless_to_type_prior(np.array([1 - q, q]), n)), 1.0 / M)
        for q in np.linspace(0.05, 0.95, 19)
    )
    assert qp <= best_ml + 1e-6, f"QP {qp} > best memoryless {best_ml}"


def test_channel_bracket_contains_exact():
    n = 6
    W = Zc
    aqp = AchievabilityQP(W, n)
    R = n * 0.3 * np.log(2)
    qp = aqp.solve_rcu_plus(R)["P_e_exact"]
    br = aqp.solve_bracketing_lp(R, kernel="exact", K=20, side="both")
    assert br["P_lo"] <= qp + 1e-4
    assert qp <= br["P_hi"] + 1e-4
    assert br["gap"] >= -1e-9


# ── rate-distortion ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("M", [4.0, 8.0])
def test_rd_bracket_straddles_exact(M):
    n = 6
    P_X = np.array([0.5, 0.5])
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    alr = AchievabilityLP_RD(P_X, d, n)
    br = alr.solve_bracketing_lp(M=M, K=48)
    D_exact = alr.exact_D_rand(br["Q_hi"], M)        # exact best-of-M at secant prior
    assert br["D_lo"] - 1e-4 <= D_exact <= br["D_hi"] + 1e-4
    assert br["gap"] >= -1e-9


# ── JSCC ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("M", [3.0, 4.0])
def test_jscc_dirac_equals_converse(M):
    P_V = np.array([0.7, 0.3])
    aj = AchievabilityJSCC(P_V, Zc, 2)
    tbj = TypeBasedJSCC(P_V, Zc, 2)
    dirac, _ = aj.solve_dirac_ramp(M)
    conv, _ = tbj.compute_converse(M)
    assert abs(dirac - conv) < 1e-6


@pytest.mark.parametrize("n", [2, 3])
def test_jscc_qp_le_memoryless_optimal(n):
    P_V = np.array([0.7, 0.3])
    aj = AchievabilityJSCC(P_V, Zc, n)
    M = float(len(P_V) ** n)            # natural operating point (L=1)
    qp, _ = aj.solve_rcu_plus(M)
    ml, _ = aj.memoryless_optimal(M)
    assert qp <= ml + 1e-6              # nesting guarantee


def test_jscc_converse_le_achievable():
    P_V = np.array([0.7, 0.3])
    n = 3
    aj = AchievabilityJSCC(P_V, Zc, n)
    tbj = TypeBasedJSCC(P_V, Zc, n)
    M = float(len(P_V) ** n)
    qp, _ = aj.solve_rcu_plus(M)
    conv, _ = tbj.compute_converse(M)
    assert conv <= qp + 1e-6


# ── excess distortion ────────────────────────────────────────────────────────
def test_excess_bracket_straddles_exact():
    n = 6
    P_X = np.array([0.5, 0.5])
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    ex = ExcessRD(P_X, d, d_th=0.0, n=n)             # excess over zero Hamming
    M = 8.0
    br = ex.solve_bracketing_lp(M=M, K=48)
    P_exact = ex.exact_P_exc(br["Q_hi"], M)
    assert br["D_lo"] - 1e-4 <= P_exact <= br["D_hi"] + 1e-4
