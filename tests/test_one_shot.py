"""
Test Suite for One-Shot Channel Coding and Rate-Distortion
===========================================================

Validates OneShotChannel and OneShotRD:
  - achievable bound vs Monte Carlo
  - LP prior optimisation + KKT conditions

Author: Nir
"""

import os, sys

import numpy as np

from fbl.one_shot_channel import OneShotChannel
from fbl.one_shot_rd import OneShotRD

from fbl.channel_achievable_utils import (
    binary_symmetric_channel, binary_erasure_channel, z_channel,
    uniform_prior, kronecker_power,
)
from fbl.achievable_utils import setup_bms_hamming


# ── shared print helper ────────────────────────────────────────────────────────

def validate_and_print(bound, prior, M_values, num_trials=1000, seed=42):
    """Validate bound for each M and print one row per M."""
    print(f"  {'M':<4} {'Theory':<12} {'MC mean':<12} {'MC std':<12} {'95% CI':<26} {'Pass'}")
    print(f"  {'-'*74}")
    for M in M_values:
        res = bound.validate(prior, M, num_trials=num_trials, seed=seed)
        ci  = f"[{res['lower_ci']:.4f}, {res['upper_ci']:.4f}]"
        ok  = "PASS" if res['within_ci'] else "FAIL"
        print(f"  {M:<4} {res['theory']:<12.6f} {res['mc_mean']:<12.6f} "
              f"{res['mc_std']:<12.6f} {ci:<26} {ok}")


# ── channel coding tests ───────────────────────────────────────────────────────

def test_channel_random():
    print("\n=== Channel: random W and Q ===")
    for x, y in [(5, 6), (5, 10), (10, 10), (10, 20)]:
        W = np.random.default_rng(0).random((x, y))
        W = W / W.sum(axis=1, keepdims=True)
        Q = np.random.default_rng(1).random(x)
        Q = Q / Q.sum()
        print(f"\n  x={x}, y={y}")
        validate_and_print(OneShotChannel(W), Q, range(2, x))


def test_channel_standard():
    print("\n=== Channel: standard channels at blocklength n ===")
    epsilon = 0.1
    channels = [
        ('BSC',  binary_symmetric_channel),
        ('BEC',  binary_erasure_channel),
        ('Z-ch', z_channel),
    ]
    for name, ch_fn in channels:
        W_single = ch_fn(epsilon)
        for n in [1, 2, 3, 4, 5]:
            W_n = kronecker_power(W_single, n)
            Q_n = uniform_prior(2 ** n)
            print(f"\n  {name}(e={epsilon})^n={n}  shape={W_n.shape}")
            validate_and_print(OneShotChannel(W_n), Q_n, range(2, n + 2))


# ── rate-distortion tests ──────────────────────────────────────────────────────

def test_rd_random():
    print("\n=== RD: random P_X, d, Q_Y ===")
    rng = np.random.default_rng(42)
    for x, y in [(5, 6), (5, 10), (10, 10), (10, 20)]:
        P_X = rng.random(x);  P_X = P_X / P_X.sum()
        Q_Y = rng.random(y);  Q_Y = Q_Y / Q_Y.sum()
        d   = rng.random((x, y))
        print(f"\n  x={x}, y={y}")
        validate_and_print(OneShotRD(P_X, d), Q_Y, range(2, min(y, 6)))


def test_rd_bms_hamming():
    print("\n=== RD: BMS(p) + Hamming distortion ===")
    p = 0.1
    for n in range(3, 7):
        P_X, d, Q_Y = setup_bms_hamming(p, n)
        print(f"\n  BMS(p={p}), n={n}  alphabet size={2**n}")
        validate_and_print(OneShotRD(P_X, d), Q_Y, range(2, 2 * n))


# ── prior optimisation tests ──────────────────────────────────────────────────

def test_channel_optimize_prior():
    print("\n=== Channel: optimize_prior vs uniform prior + KKT check ===")
    channels = [
        ('BSC(0.1)',  binary_symmetric_channel(0.1)),
        ('Z-ch(0.1)', z_channel(0.1)),
    ]
    for name, W in channels:
        bound  = OneShotChannel(W)
        Q_unif = uniform_prior(W.shape[0])
        print(f"\n  {name}")
        print(f"  {'M':<4} {'Opt':<12} {'Unif':<12} {'KKT c1':<8} {'KKT c2':<8} {'Prior'}")
        for M in [2, 4, 8, 16]:
            Q_opt, metric_opt = bound.optimize_prior(M)
            kkt               = bound.check_kkt(M, Q_opt, bound._s_dual)
            curve_unif        = bound.compute_curve(Q_unif)
            metric_unif       = bound.theory(curve_unif, M)
            c1 = "PASS" if kkt['cond1'] else "FAIL"
            c2 = "PASS" if kkt['cond2'] else "FAIL"
            print(f"  {M:<4} {metric_opt:<12.6f} {metric_unif:<12.6f} {c1:<8} {c2:<8} {np.round(Q_opt, 4)}")


def test_rd_optimize_prior():
    print("\n=== RD: optimize_prior vs uniform prior + KKT check ===")
    p = 0.1
    for n in [1, 2, 3]:
        P_X, d, Q_unif = setup_bms_hamming(p, n)
        bound = OneShotRD(P_X, d)
        print(f"\n  BMS(p={p}), n={n}")
        print(f"  {'M':<4} {'Opt':<12} {'Unif':<12} {'KKT c1':<8} {'KKT c2'}")
        for M in [2, 4, 8]:
            Q_opt, metric_opt = bound.optimize_prior(M)
            kkt               = bound.check_kkt(M, Q_opt, bound._s_dual)
            curve_unif        = bound.compute_curve(Q_unif)
            metric_unif       = bound.theory(curve_unif, M)
            c1 = "PASS" if kkt['cond1'] else "FAIL"
            c2 = "PASS" if kkt['cond2'] else "FAIL"
            print(f"  {M:<4} {metric_opt:<12.6f} {metric_unif:<12.6f} {c1:<8} {c2}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_channel_random()
    test_channel_standard()
    test_rd_random()
    test_rd_bms_hamming()
    test_channel_optimize_prior()
    test_rd_optimize_prior()
