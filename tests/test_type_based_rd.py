"""
Test Suite for Type-Based Rate-Distortion
==========================================

Validates TypeBasedRD against OneShotRD:

  1. A-curve: for any type prior P_T_Y, the type-based A-curve must exactly
     match the one-shot A-curve built with the equivalent sequence prior.
     Tested for: uniform, memoryless, and random type priors.

  2. optimize_prior at n=1: type-based and one-shot LPs must give the same
     optimal metric (at n=1 the two search spaces are identical).

  3. optimize_prior at n>1: type-based metric >= one-shot metric (the
     type-based LP is a restricted search over type-constant priors, and
     since both minimise distortion the type-based optimum is no better).

Author: Nir
"""

import os, sys

import numpy as np
from fbl.F_curve import compare_F_curves

from fbl.type_based_rd import TypeBasedRD
from fbl.type_class_core import composition_count
from fbl.type_based_utils import (
    type_prior_to_one_shot,
    uniform_type_prior,
    memoryless_to_type_prior,
    random_type_prior,
)


# ── source/distortion helpers ──────────────────────────────────────────────────

def bms_hamming(p):
    """Binary memoryless source with Hamming distortion."""
    P_X = np.array([1 - p, p])
    d   = np.array([[0.0, 1.0], [1.0, 0.0]])
    return P_X, d


def random_source_distortion(k_x, k_y, seed=0):
    """Random source distribution and distortion matrix."""
    rng = np.random.default_rng(seed)
    P_X = rng.random(k_x);  P_X /= P_X.sum()
    d   = rng.random((k_x, k_y))
    return P_X, d


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_priors(n, k_y, seed=0):
    """Return uniform, memoryless, and random type priors for given n, k_y."""
    rng = np.random.default_rng(seed)
    Q_single = rng.random(k_y);  Q_single /= Q_single.sum()
    return {
        'uniform':    uniform_type_prior(n, k_y),
        'memoryless': memoryless_to_type_prior(Q_single, n),
        'random':     random_type_prior(n, k_y, seed=seed),
    }


# ── A-curve validation ─────────────────────────────────────────────────────────

def _check_a_curve(P_X_single, d_single, n_values, label=''):
    """
    For each n, verify type-based A-curve == one-shot A-curve for three
    prior types (uniform, memoryless, random).
    """
    k_y = d_single.shape[1]
    print(f"\n  {label}")
    print(f"  {'n':<4} {'prior':<12} {'result'}")
    print(f"  {'-'*30}")

    for n in n_values:
        tb       = TypeBasedRD(P_X_single, d_single, n)
        one_shot = tb.get_one_shot_object()
        priors   = _make_priors(n, k_y)

        for prior_name, P_T_Y in priors.items():
            curve_tb = tb.compute_curve(P_T_Y)
            Q_seq    = tb.prior_to_one_shot(P_T_Y)
            curve_os = one_shot.compute_curve(Q_seq)

            res = compare_F_curves(*curve_tb, *curve_os)
            ok  = "PASS" if res['all_close'] else f"FAIL (max_diff={res['max_abs_diff']:.2e})"
            print(f"  {n:<4} {prior_name:<12} {ok}")
            assert res['all_close'], \
                f"A-curve mismatch: {label}, n={n}, prior={prior_name}"


# ── LP validation ──────────────────────────────────────────────────────────────

def _check_optimize_prior_n1(P_X_single, d_single, label=''):
    """
    At n=1, type-based and one-shot LPs search the same space.
    Optimal metrics must agree.
    """
    n  = 1
    tb = TypeBasedRD(P_X_single, d_single, n)
    os = tb.get_one_shot_object()

    print(f"\n  {label}  (n=1 LP comparison)")
    print(f"  {'M':<4} {'type-based':<14} {'one-shot':<14} {'diff':<10} {'result'}")
    print(f"  {'-'*56}")

    for M in [2, 4, 8, 16]:
        _, metric_tb = tb.optimize_prior(M)
        _, metric_os = os.optimize_prior(M)
        diff = abs(metric_tb - metric_os)
        ok   = "PASS" if diff < 1e-5 else "FAIL"
        print(f"  {M:<4} {metric_tb:<14.8f} {metric_os:<14.8f} {diff:<10.2e} {ok}")
        assert diff < 1e-5, \
            f"LP mismatch at n=1: {label}, M={M}, diff={diff:.2e}"


def _check_optimize_prior_n_gt1(P_X_single, d_single, n_values, label=''):
    """
    For n>1, type-based LP metric >= one-shot LP metric.
    (Both minimise distortion; type-based is restricted so its optimum is
    no better than the unconstrained one-shot optimum.)
    """
    print(f"\n  {label}  (n>1: type-based >= one-shot)")
    print(f"  {'n':<4} {'M':<4} {'type-based':<14} {'one-shot':<14} {'gap':<10} {'result'}")
    print(f"  {'-'*60}")

    for n in n_values:
        tb = TypeBasedRD(P_X_single, d_single, n)
        os = tb.get_one_shot_object()
        for M in [2, 4, 8]:
            _, metric_tb = tb.optimize_prior(M)
            _, metric_os = os.optimize_prior(M)
            gap = metric_tb - metric_os
            ok  = "PASS" if gap >= -1e-5 else "FAIL"
            print(f"  {n:<4} {M:<4} {metric_tb:<14.8f} {metric_os:<14.8f} {gap:<10.2e} {ok}")
            assert gap >= -1e-5, \
                f"Type-based < one-shot: {label}, n={n}, M={M}, gap={gap:.2e}"


# ── test runners ───────────────────────────────────────────────────────────────

def test_a_curve_suite():
    print("\n=== A-curve: type-based == one-shot ===")
    sources = [
        ('BMS(0.1)+Hamming', *bms_hamming(0.1)),
        ('BMS(0.3)+Hamming', *bms_hamming(0.3)),
    ]
    for label, P_X, d in sources:
        _check_a_curve(P_X, d, n_values=range(1, 7), label=label)

    random_sources = [
        ('random(2x2,s=0)', *random_source_distortion(2, 2, seed=0)),
        ('random(2x3,s=1)', *random_source_distortion(2, 3, seed=1)),
        ('random(3x3,s=2)', *random_source_distortion(3, 3, seed=2)),
    ]
    for label, P_X, d in random_sources:
        _check_a_curve(P_X, d, n_values=range(1, 4), label=label)


def test_lp_n1_suite():
    print("\n=== LP at n=1: type-based == one-shot ===")
    sources = [
        ('BMS(0.1)+Hamming', *bms_hamming(0.1)),
        ('BMS(0.3)+Hamming', *bms_hamming(0.3)),
        ('random(2x2,s=0)',  *random_source_distortion(2, 2, seed=0)),
        ('random(2x3,s=1)',  *random_source_distortion(2, 3, seed=1)),
        ('random(3x3,s=2)',  *random_source_distortion(3, 3, seed=2)),
    ]
    for label, P_X, d in sources:
        _check_optimize_prior_n1(P_X, d, label=label)


def test_lp_n_gt1_suite():
    print("\n=== LP at n>1: type-based >= one-shot ===")
    sources = [
        ('BMS(0.1)+Hamming', *bms_hamming(0.1)),
        ('random(2x2,s=0)',  *random_source_distortion(2, 2, seed=0)),
        ('random(2x3,s=1)',  *random_source_distortion(2, 3, seed=1)),
        ('random(3x3,s=2)',  *random_source_distortion(3, 3, seed=2)),
    ]
    for label, P_X, d in sources:
        _check_optimize_prior_n_gt1(P_X, d, n_values=[2, 3], label=label)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_a_curve_suite()
    test_lp_n1_suite()
    test_lp_n_gt1_suite()
    print("\nAll tests passed.")
