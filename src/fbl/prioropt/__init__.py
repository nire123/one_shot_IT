"""
fbl.prioropt — prior optimization of finite-blocklength bounds
==============================================================

The **Φ-view** unifies prior optimization across all settings: the bound is
``J = c^T Φ(A Q)`` over a product of simplices, and the same first-order
**simplex march** (KKT/FW-gap certified) optimizes it for any kernel.

- :mod:`fbl.prioropt.phi_view`     — the relaxation ``J = c^T Φ(A Q)``: the
  potentials Φ (and Φ′, κ), the literal ``(A, c)`` preprocess (one-shot), and the
  type-based evaluators (``J_typebased_{channel,rd,jscc}``).
- :mod:`fbl.prioropt.phi_simplex`  — the achievability optimizer: ``build_program``
  (channel / rd / jscc), ``optimize`` (projected gradient / Frank–Wolfe),
  ``check_kkt`` (water-filling certificate).

Exact convex solvers, kept as validation anchors / staircase builders for the
march:

- :class:`AchievabilityQP`     — channel, exact QP (RCU⁺) + bracketing LP
- :class:`AchievabilityLP_RD`  — rate-distortion, bracketing LP + ``exact_D_rand``
- :class:`AchievabilityJSCC`   — JSCC, exact QP (L=1) + bracketing LP + baselines
- :class:`ExcessRD`            — excess distortion (indicator-distortion wrapper)

The standard *memoryless* baseline is "the optimal single-letter prior applied
i.i.d." (see ``AchievabilityJSCC.memoryless_optimal``).
"""

from fbl.prioropt.achievability_qp import AchievabilityQP
from fbl.prioropt.achievability_lp_rd import AchievabilityLP_RD
from fbl.prioropt.achievability_jscc import AchievabilityJSCC
from fbl.prioropt.excess_distortion_rd import ExcessRD
from fbl.type_based_utils import rcu_plus_from_F_curve, marginal_input
from fbl.prioropt import phi_view, phi_simplex

__all__ = [
    "AchievabilityQP", "AchievabilityLP_RD", "AchievabilityJSCC", "ExcessRD",
    "rcu_plus_from_F_curve", "marginal_input", "phi_view", "phi_simplex",
]
