"""
fbl.prioropt — prior optimization of finite-blocklength bounds
==============================================================

Achievability prior optimization (the exact convex program of the PEP
framework) and converse prior optimization (LP), type-based:

- :class:`AchievabilityQP`      — channel, exact QP (RCU^+ kernel) + bracketing LP
- :class:`AchievabilityLP_RD`   — rate-distortion, bracketing LP (degree-M kernel)
- :class:`AchievabilityJSCC`    — JSCC, exact QP (L=1) + bracketing LP + ``memoryless_optimal``
- :class:`ExcessRD`             — excess-distortion prior optimization
- :class:`TypeBasedBlockLP`     — channel converse-prior LP (block / chord rule)
- :class:`TypeBasedBlockLPRD`   — RD converse-prior LP

The standard *memoryless* baseline is "the optimal single-letter prior applied
i.i.d." — computed by the exact n=1 program and extended (see
``AchievabilityJSCC.memoryless_optimal``).
"""

from fbl.prioropt.achievability_qp import AchievabilityQP
from fbl.prioropt.achievability_lp_rd import AchievabilityLP_RD
from fbl.prioropt.achievability_jscc import AchievabilityJSCC
from fbl.prioropt.excess_distortion_rd import ExcessRD
from fbl.prioropt.typebased_block_lp import TypeBasedBlockLP, rcu_plus_from_F_curve
from fbl.prioropt.typebased_block_lp_rd import TypeBasedBlockLPRD
from fbl.prioropt.direct_program import DirectPriorOpt

__all__ = [
    "AchievabilityQP", "AchievabilityLP_RD", "AchievabilityJSCC", "ExcessRD",
    "TypeBasedBlockLP", "TypeBasedBlockLPRD", "rcu_plus_from_F_curve",
    "DirectPriorOpt",
]
