"""
Test Suite for Type-Based Channel Coding
==========================================

Validates TypeBasedChannel against OneShotChannel:

  1. F-curve: for any type prior P_T_X, the type-based F-curve must exactly
     match the one-shot F-curve built with the equivalent sequence prior.
     Tested for: uniform, memoryless, and random type priors.

  2. optimize_prior at n=1: type-based and one-shot LPs must give the same
     optimal metric (at n=1 the two search spaces are identical).

  3. optimize_prior at n>1: type-based metric >= one-shot metric (the
     type-based LP is a restricted search over type-constant priors).

Author: Nir
"""

import os, sys

import numpy as np
from fbl.F_curve import compare_F_curves

from fbl.type_based_channel import TypeBasedChannel
from fbl.type_class_core import composition_count
from fbl.type_based_utils import (
    type_prior_to_one_shot,
    uniform_type_prior,
    memoryless_to_type_prior,
    random_type_prior,
)
from fbl.channel_achievable_utils import (
    binary_symmetric_channel,
    binary_erasure_channel,
    z_channel,
)


def random_channel(k_x, k_y, seed=0):
    """Random row-stochastic channel matrix of shape (k_x, k_y)."""
    rng = np.random.default_rng(seed)
    W = rng.random((k_x, k_y))
    W /= W.sum(axis=1, keepdims=True)
    return W


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_priors(n, k_x, seed=0):
    """Return uniform, memoryless, and random type priors for given n, k_x."""
    rng = np.random.default_rng(seed)
    Q_single = rng.random(k_x);  Q_single /= Q_single.sum()
    return {
        'uniform':    uniform_type_prior(n, k_x),
        'memoryless': memoryless_to_type_prior(Q_single, n),
        'random':     random_type_prior(n, k_x, seed=seed),
    }


# ── F-curve validation ─────────────────────────────────────────────────────────

def _check_f_curve(W_single, n_values, label=''):
    """
    For each n, verify type-based F-curve == one-shot F-curve for three
    prior types (uniform, memoryless, random).
    """
    k_x = W_single.shape[0]
    print(f"\n  {label}")
    print(f"  {'n':<4} {'prior':<12} {'result'}")
    print(f"  {'-'*30}")

    for n in n_values:
        tb       = TypeBasedChannel(W_single, n)
        one_shot = tb.get_one_shot_object()
        priors   = _make_priors(n, k_x)

        for prior_name, P_T_X in priors.items():
            curve_tb = tb.compute_curve(P_T_X)
            Q_seq    = tb.prior_to_one_shot(P_T_X)
            curve_os = one_shot.compute_curve(Q_seq)

            res = compare_F_curves(*curve_tb, *curve_os)
            ok  = "PASS" if res['all_close'] else f"FAIL (max_diff={res['max_abs_diff']:.2e})"
            print(f"  {n:<4} {prior_name:<12} {ok}")
            assert res['all_close'], \
                f"F-curve mismatch: {label}, n={n}, prior={prior_name}"


# ── LP validation ──────────────────────────────────────────────────────────────

def _check_optimize_prior_n1(W_single, label=''):
    """
    At n=1, type-based and one-shot LPs search the same space.
    Optimal metrics must agree.
    """
    n  = 1
    tb = TypeBasedChannel(W_single, n)
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


def _check_optimize_prior_n_gt1(W_single, n_values, label=''):
    """
    For n>1, type-based LP >= one-shot LP (restricted search space).
    """
    print(f"\n  {label}  (n>1: type-based >= one-shot)")
    print(f"  {'n':<4} {'M':<4} {'type-based':<14} {'one-shot':<14} {'gap':<10} {'result'}")
    print(f"  {'-'*60}")

    for n in n_values:
        tb = TypeBasedChannel(W_single, n)
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

def test_f_curve_suite():
    print("\n=== F-curve: type-based == one-shot ===")
    channels = [
        ('BSC(0.1)',  binary_symmetric_channel(0.1)),
        ('BEC(0.1)',  binary_erasure_channel(0.1)),
        ('Z-ch(0.1)', z_channel(0.1)),
    ]
    for label, W in channels:
        _check_f_curve(W, n_values=range(1, 7), label=label)

    random_channels = [
        ('random(2x3,s=0)', random_channel(2, 3, seed=0)),
        ('random(3x4,s=1)', random_channel(3, 4, seed=1)),
        ('random(4x5,s=2)', random_channel(4, 5, seed=2)),
    ]
    for label, W in random_channels:
        _check_f_curve(W, n_values=range(1, 4), label=label)


def test_lp_n1_suite():
    print("\n=== LP at n=1: type-based == one-shot ===")
    channels = [
        ('BSC(0.1)',  binary_symmetric_channel(0.1)),
        ('BEC(0.1)',  binary_erasure_channel(0.1)),
        ('Z-ch(0.1)', z_channel(0.1)),
    ]
    for label, W in channels:
        _check_optimize_prior_n1(W, label=label)

    random_channels = [
        ('random(2x3,s=0)', random_channel(2, 3, seed=0)),
        ('random(3x4,s=1)', random_channel(3, 4, seed=1)),
        ('random(4x5,s=2)', random_channel(4, 5, seed=2)),
    ]
    for label, W in random_channels:
        _check_optimize_prior_n1(W, label=label)


def test_converse_rate_at_eps_single_lp():
    """The single-LP converse rate inversion (min w s.t. success >= 1-eps) must
    match a bisection that solves the fixed-M converse LP per step."""
    eps, ln2 = 1e-3, np.log(2.0)
    for n in (6, 10):
        tbc = TypeBasedChannel(z_channel(0.1), n)
        r_lp = tbc.converse_rate_at_eps(eps) / ln2              # bits/sym
        lo, hi = 0.0, 1.2
        for _ in range(24):                                    # bisection reference
            mid = 0.5 * (lo + hi)
            err = tbc.optimize_prior(float(np.exp(n * mid * ln2)))[1]
            lo, hi = (mid, hi) if err <= eps else (lo, mid)
        assert abs(r_lp - lo) <= 1e-4, f"n={n}: single-LP {r_lp} vs bisection {lo}"


def test_lp_n_gt1_suite():
    print("\n=== LP at n>1: type-based >= one-shot ===")
    channels = [
        ('BSC(0.1)',  binary_symmetric_channel(0.1)),
        ('Z-ch(0.1)', z_channel(0.1)),
    ]
    for label, W in channels:
        _check_optimize_prior_n_gt1(W, n_values=[2, 3, 4], label=label)

    random_channels = [
        ('random(2x3,s=0)', random_channel(2, 3, seed=0)),
        ('random(3x4,s=1)', random_channel(3, 4, seed=1)),
        ('random(4x5,s=2)', random_channel(4, 5, seed=2)),
    ]
    for label, W in random_channels:
        _check_optimize_prior_n_gt1(W, n_values=[2, 3], label=label)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_f_curve_suite()
    test_lp_n1_suite()
    test_lp_n_gt1_suite()
    print("\nAll tests passed.")
