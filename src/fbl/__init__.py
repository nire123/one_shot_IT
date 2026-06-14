"""
fbl — Finite-Blocklength bounds via the PEP error-spectrum framework
====================================================================

A small, validated library for one-shot / finite-blocklength information theory
across three settings — **channel coding**, **rate-distortion (RD)**, and
**joint source-channel coding (JSCC)** — in two implementation flavours:

- **one-shot**: exact, in the lifted ``|X|^n`` product space (ground truth, +MC);
- **type-based**: polynomial in ``n`` via the method of types.

For each setting the library computes the **achievable** (random-coding) bound,
the **converse** (meta-converse LP), and — the distinguishing feature — the
**prior-optimised** versions of both: the converse prior via an LP, and the
*achievability* prior via the exact convex program (QP / bracketing LP) of the
PEP framework.

Public API
----------
Settings (one-shot):      :class:`OneShotChannel`, :class:`OneShotRD`, :class:`OneShotJSCC`
Settings (type-based):    :class:`TypeBasedChannel`, :class:`TypeBasedRD`, :class:`TypeBasedJSCC`
Prior optimization:       see :mod:`fbl.prioropt`
"""

from fbl.one_shot_channel import OneShotChannel
from fbl.one_shot_rd import OneShotRD
from fbl.one_shot_jscc import OneShotJSCC
from fbl.type_based_channel import TypeBasedChannel
from fbl.type_based_rd import TypeBasedRD
from fbl.type_based_jscc import TypeBasedJSCC

__all__ = [
    "OneShotChannel", "OneShotRD", "OneShotJSCC",
    "TypeBasedChannel", "TypeBasedRD", "TypeBasedJSCC",
]
__version__ = "0.1.0"
