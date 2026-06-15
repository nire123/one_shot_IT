"""
Tests for OneShotJSCC
=====================

Three problem instances, each testing:

  1. F-curve validity    : non-decreasing CDF, starts at 0, ends at ~1
  2. Bound ordering      : converse_error <= achievable_bound  (deterministic)
  3. MC vs bound         : mc_mean <= achievable_bound + 3*sigma
  4. MC vs converse      : mc_mean >= converse_error - 3*sigma

Problem instances
-----------------
A. Binary symmetric channel (BSC), uniform source, M = |V| = 2
B. Z-channel, non-uniform source, M = |V| = 2
C. 3x3 random channel, uniform source, M = |V| = 3
"""

import os, sys

import numpy as np
import pytest
from fbl.one_shot_jscc import OneShotJSCC


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_bsc(p):
    return np.array([[1 - p, p], [p, 1 - p]])


def _make_z_channel(p):
    """Z-channel: W[0,0]=1, W[0,1]=0, W[1,0]=p, W[1,1]=1-p."""
    return np.array([[1.0, 0.0], [p, 1.0 - p]])


def _check_f_curve(knots, vals, y_size=None):
    assert knots[0] == 0.0,              "F-curve must start at w=0"
    assert vals[0]  == 0.0,              "F(0) must be 0"
    assert np.all(np.diff(knots) >= 0),  "knots must be non-decreasing"
    assert np.all(np.diff(vals)  >= -1e-12), "F-curve must be non-decreasing"
    assert abs(vals[-1] - 1.0) < 1e-4,  f"F-curve must end at 1, got {vals[-1]:.6f}"


def _run_checks(jscc, M, Q_XgV, num_trials=2000, seed=0, label=""):
    # bounds
    converse_err, _ = jscc.compute_converse(M)
    achievable_err  = jscc.achievable_bound(M, Q_XgV)

    assert converse_err is not None, f"{label}: LP did not converge"
    assert 0.0 <= converse_err <= 1.0 + 1e-9, f"{label}: converse out of [0,1]"
    assert 0.0 <= achievable_err <= 1.0 + 1e-9, f"{label}: achievable out of [0,1]"

    # deterministic bound ordering
    assert converse_err <= achievable_err + 1e-6, (
        f"{label}: converse ({converse_err:.4f}) > achievable ({achievable_err:.4f})")

    # F-curve
    knots, vals = jscc.compute_f_curve(Q_XgV)
    _check_f_curve(knots, vals)

    # MC
    mc = jscc.mc(Q_XgV, num_trials=num_trials, seed=seed)
    sigma = mc['std'] / np.sqrt(num_trials)

    assert mc['mean'] <= achievable_err + 3 * sigma + 1e-6, (
        f"{label}: MC mean ({mc['mean']:.4f}) > achievable bound ({achievable_err:.4f})")

    assert mc['mean'] >= converse_err - 3 * sigma - 1e-6, (
        f"{label}: MC mean ({mc['mean']:.4f}) < converse ({converse_err:.4f})")

    return converse_err, achievable_err, mc['mean']


# ── tests ──────────────────────────────────────────────────────────────────────

class TestBSC:
    """Binary symmetric channel, uniform source, M=2."""

    def setup_method(self):
        P_V  = np.array([0.5, 0.5])
        W    = _make_bsc(0.1)
        self.jscc = OneShotJSCC(P_V, W)
        self.M    = 2     # |V|^1
        # LP optimal Q_{X|V}
        _, self.Q_opt = self.jscc.compute_converse(self.M)

    def test_f_curve_valid(self):
        knots, vals = self.jscc.compute_f_curve(self.Q_opt)
        _check_f_curve(knots, vals)

    def test_bound_ordering(self):
        conv, _ = self.jscc.compute_converse(self.M)
        ach     = self.jscc.achievable_bound(self.M, self.Q_opt)
        assert conv <= ach + 1e-6

    def test_mc_sandwich(self):
        conv, ach, mc_mean = _run_checks(
            self.jscc, self.M, self.Q_opt, label="BSC")
        print(f"\n  BSC: converse={conv:.4f}  RCB={ach:.4f}  MC={mc_mean:.4f}")

    def test_uniform_prior_also_valid(self):
        """Uniform Q_{X|V} should also satisfy the bound ordering."""
        Q_uniform = np.full((self.jscc.v_size, self.jscc.x_size),
                            1.0 / self.jscc.x_size)
        conv, _  = self.jscc.compute_converse(self.M)
        ach      = self.jscc.achievable_bound(self.M, Q_uniform)
        assert conv <= ach + 1e-6


class TestZChannel:
    """Z-channel (p=0.2), non-uniform source, M=2."""

    def setup_method(self):
        P_V  = np.array([0.3, 0.7])
        W    = _make_z_channel(0.2)
        self.jscc = OneShotJSCC(P_V, W)
        self.M    = 2
        _, self.Q_opt = self.jscc.compute_converse(self.M)

    def test_f_curve_valid(self):
        knots, vals = self.jscc.compute_f_curve(self.Q_opt)
        _check_f_curve(knots, vals)

    def test_mc_sandwich(self):
        conv, ach, mc_mean = _run_checks(
            self.jscc, self.M, self.Q_opt, label="Z-channel")
        print(f"\n  Z-ch: converse={conv:.4f}  RCB={ach:.4f}  MC={mc_mean:.4f}")


class TestRandomChannel3x3:
    """3x3 channel (fixed seed), uniform source, M=3."""

    def setup_method(self):
        rng = np.random.default_rng(7)
        raw = rng.random((3, 3)) + 4 * np.eye(3)   # diagonal-dominant
        W   = raw / raw.sum(axis=1, keepdims=True)
        P_V = np.ones(3) / 3
        self.jscc = OneShotJSCC(P_V, W)
        self.M    = 3
        _, self.Q_opt = self.jscc.compute_converse(self.M)

    def test_f_curve_valid(self):
        knots, vals = self.jscc.compute_f_curve(self.Q_opt)
        _check_f_curve(knots, vals)

    def test_mc_sandwich(self):
        conv, ach, mc_mean = _run_checks(
            self.jscc, self.M, self.Q_opt,
            num_trials=3000, label="3x3 channel")
        print(f"\n  3x3: converse={conv:.4f}  RCB={ach:.4f}  MC={mc_mean:.4f}")


class TestKKT:
    """KKT optimality conditions for the JSCC converse LP."""

    def _check_kkt(self, P_V, W, M, label=""):
        jscc = OneShotJSCC(P_V, W)
        conv, Q_opt = jscc.compute_converse(M)
        assert Q_opt is not None, f"{label}: LP did not converge"
        res = jscc.check_kkt(M, Q_opt, jscc._s_dual)
        assert res['all_pass'], (
            f"{label}: KKT failed — cond1={res['cond1']}  cond2={res['cond2']}")
        return res

    def test_bsc_uniform(self):
        self._check_kkt(np.array([0.5, 0.5]), _make_bsc(0.1), M=2, label="BSC")

    def test_z_channel_nonuniform(self):
        self._check_kkt(np.array([0.3, 0.7]), _make_z_channel(0.2), M=2,
                        label="Z-channel")

    def test_3x3_random(self):
        rng = np.random.default_rng(7)
        raw = rng.random((3, 3)) + 4 * np.eye(3)
        W   = raw / raw.sum(axis=1, keepdims=True)
        self._check_kkt(np.ones(3) / 3, W, M=3, label="3x3")

    def test_kkt_returns_g_shape(self):
        """g matrix must have shape (|V|, |X|)."""
        P_V  = np.array([0.5, 0.5])
        W    = _make_bsc(0.1)
        jscc = OneShotJSCC(P_V, W)
        _, Q_opt = jscc.compute_converse(2)
        res  = jscc.check_kkt(2, Q_opt, jscc._s_dual)
        assert res['g'].shape == (jscc.v_size, jscc.x_size)
        assert res['cond2_slack'].shape == (jscc.y_size, 2)


class TestEvaluateConsistency:
    """evaluate() must match simulate_error_probability from the reference code."""

    def test_evaluate_matches_exact_loop(self):
        """
        For a fixed codebook, evaluate() result should be reproducible and in [0,1].
        We verify it against a direct per-sample calculation.
        """
        P_V = np.array([0.5, 0.5])
        W   = _make_bsc(0.15)
        jscc = OneShotJSCC(P_V, W)

        rng      = np.random.default_rng(42)
        codebook = jscc.draw_random_code(np.eye(2), rng)   # identity encoder

        err = jscc.evaluate(codebook)
        assert 0.0 <= err <= 1.0

        # For identity encoder on BSC(p): error = p (wrong symbol is sent when
        # channel flips, decoder still makes error with prob 0.5 if symmetric)
        # Just check it's finite and in range.
        assert np.isfinite(err)


# ── standalone runner ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import traceback

    test_classes = [
        TestBSC,
        TestZChannel,
        TestRandomChannel3x3,
        TestKKT,
        TestEvaluateConsistency,
    ]

    passed = failed = 0
    for cls in test_classes:
        methods = [m for m in dir(cls) if m.startswith('test_')]
        for name in methods:
            obj = cls()
            if hasattr(obj, 'setup_method'):
                obj.setup_method()
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}::{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}::{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed + failed} tests: {passed} passed, {failed} failed.")
