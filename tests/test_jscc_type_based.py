"""
Tests for TypeBasedJSCC
========================

Key contract
------------
1. F-curve validity    : non-decreasing, knots in [0,1], vals in [0,1]
2. n=1 exact match     : type-based F-curve and bounds == one-shot for same encoder
3. n>1 converse match  : type-based LP converse == one-shot LP converse on n-letter
                         product system (BSC+uniform source — optimal encoder IS
                         type-constant by symmetry)
4. n>1 achievable match: type-based achievable bound (with correct prior) == one-shot
                         achievable bound with the type-constant encoder converted to
                         a sequence-level distribution
5. Bound ordering      : converse ≤ achievable for all n

Prior normalisation note
------------------------
compute_converse() returns Q_cond: a CONDITIONAL distribution where each T_V block
sums to 1.  compute_f_curve() / achievable_bound() need a prior where each T_V block
sums to 1/k_v^n (so that per-output-type knot total = 1).
Correct prior = Q_cond / k_v^n.

Channels tested
---------------
A. BSC(0.1),  P_V = [0.5, 0.5],  n = 1, 2, 3
B. Z-channel(0.2), P_V = [0.3, 0.7],  n = 1
C. 3×3 random channel, P_V = uniform,  n = 1
"""

import os, sys

import numpy as np
from fbl.one_shot_jscc import OneShotJSCC
from fbl.type_based_jscc import TypeBasedJSCC


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_bsc(p):
    return np.array([[1 - p, p], [p, 1 - p]])


def _make_z_channel(p):
    return np.array([[1.0, 0.0], [p, 1.0 - p]])


def _kronecker_power(W, n):
    """W^{⊗n}: (k_x^n, k_y^n) product channel."""
    result = W
    for _ in range(n - 1):
        result = np.kron(result, W)
    return result


def _product_source(P_V, n):
    """P_V^{⊗n}: joint distribution over k_v^n sequences."""
    result = P_V
    for _ in range(n - 1):
        result = np.kron(result, P_V)
    return result


def _type_constant_to_sequence_encoder(Q_cond, tb):
    """
    Convert a type-constant conditional Q_cond (cond_vx.len, tuple-indexed,
    each T_V block sums to 1) to a full sequence-level encoder matrix
    Q_seq of shape (k_v^n, k_x^n).

    Q_seq[v_idx, x_idx] = P(x^n | v^n) for the type-constant encoder.

    For a type-constant encoder:
        P(x^n | v^n) = Q_cond[ tuple_idx(T_V(v^n), T_{X|V}(v^n,x^n)) ]
                       / |conditional-type-class of x^n given T_V(v^n)|

    We need to enumerate all (v^n, x^n) pairs, look up their joint type,
    and assign probabilities.
    """
    from fbl.type_class_core import enumerate_type_class, composition_to_index

    n    = tb.n
    k_v  = tb.k_v
    k_x  = tb.k_x
    kv_n = k_v ** n
    kx_n = k_x ** n

    Q_seq = np.zeros((kv_n, kx_n))

    # Build a lookup: for each joint type T_VX (in cond_vx tuple ordering)
    # what is the conditional-type-class size?
    cond_size = {}   # tuple_idx -> |T_{X|V}| (class of x^n sequences with that type)
    for i_T_v, T_v, i_T_xgv, T_vx, log_cond_size in tb._cond_vx.enumerate():
        idx = tb._cond_vx.tuple_2_ix(i_T_v, i_T_xgv)
        cond_size[idx] = int(round(np.exp(log_cond_size)))

    def seq_to_type(seq_flat, k):
        """Sequence (base-k integer) -> type (count vector)."""
        T = np.zeros(k, dtype=int)
        for _ in range(n):
            T[seq_flat % k] += 1
            seq_flat //= k
        return T

    from fbl.type_class_core import matrix_to_index

    for v_idx in range(kv_n):
        T_v = seq_to_type(v_idx, k_v)
        i_T_v = composition_to_index(T_v)
        for x_idx in range(kx_n):
            T_x = seq_to_type(x_idx, k_x)
            # Build T_vx: for each symbol position, count (v,x) pairs
            T_vx = np.zeros((k_v, k_x), dtype=int)
            v_tmp, x_tmp = v_idx, x_idx
            for _ in range(n):
                v_sym = v_tmp % k_v
                x_sym = x_tmp % k_x
                T_vx[v_sym, x_sym] += 1
                v_tmp //= k_v
                x_tmp //= k_x
            # Conditional type index
            i_T_xgv = matrix_to_index(T_vx)
            tuple_idx = tb._cond_vx.tuple_2_ix(i_T_v, i_T_xgv)
            Q_seq[v_idx, x_idx] = Q_cond[tuple_idx] / cond_size[tuple_idx]

    return Q_seq


def _check_f_curve(knots, vals, label=""):
    assert knots[0] == 0.0,                f"{label}: knots[0] != 0"
    assert vals[0]  == 0.0,                f"{label}: vals[0] != 0"
    assert np.all(np.diff(knots) >= -1e-12), f"{label}: knots not non-decreasing"
    assert np.all(np.diff(vals)  >= -1e-9),  f"{label}: vals not non-decreasing"
    assert abs(vals[-1] - 1.0) < 1e-3,    f"{label}: vals[-1]={vals[-1]:.6f} != 1"
    assert knots[-1] <= 1.0 + 1e-6,       f"{label}: knots[-1]={knots[-1]:.6f} > 1"


def _compare_curves(k1, v1, k2, v2, label="", tol=1e-4):
    """Check two F-curves agree everywhere on the union of their knots."""
    all_knots = np.union1d(k1, k2)
    f1 = np.interp(all_knots, k1, v1)
    f2 = np.interp(all_knots, k2, v2)
    max_diff = np.max(np.abs(f1 - f2))
    assert max_diff < tol, (
        f"{label}: max curve diff = {max_diff:.2e} (tol={tol:.1e})")


# ── n=1 tests ─────────────────────────────────────────────────────────────────

class TestBSC_n1:
    """BSC(0.1), uniform source, n=1, k_v=2."""

    def setup_method(self):
        self.P_V = np.array([0.5, 0.5])
        self.W   = _make_bsc(0.1)
        self.M   = 2.0
        self.tb  = TypeBasedJSCC(self.P_V, self.W, n=1)
        self.os  = OneShotJSCC(self.P_V, self.W)
        _, self.Q_opt = self.os.compute_converse(self.M)
        # correct type prior: q_xgv_to_type_prior divides by k_v (= k_v^1)
        self.P_T_VX = self.tb.q_xgv_to_type_prior(self.Q_opt)

    def test_prior_sums_to_one(self):
        assert abs(self.P_T_VX.sum() - 1.0) < 1e-9

    def test_f_curve_valid(self):
        k, v = self.tb.compute_f_curve(self.P_T_VX)
        _check_f_curve(k, v, label="BSC n=1")

    def test_f_curve_matches_one_shot(self):
        k_tb, v_tb = self.tb.compute_f_curve(self.P_T_VX)
        k_os, v_os = self.os.compute_f_curve(self.Q_opt)
        _compare_curves(k_tb, v_tb, k_os, v_os, label="BSC n=1")

    def test_achievable_matches_one_shot(self):
        ach_tb = self.tb.achievable_bound(self.M, self.P_T_VX)
        ach_os = self.os.achievable_bound(self.M, self.Q_opt)
        assert abs(ach_tb - ach_os) < 1e-4, \
            f"type={ach_tb:.6f}  oneshot={ach_os:.6f}"

    def test_converse_matches_one_shot(self):
        conv_tb, _ = self.tb.compute_converse(self.M)
        conv_os, _ = self.os.compute_converse(self.M)
        assert abs(conv_tb - conv_os) < 1e-4, \
            f"type={conv_tb:.6f}  oneshot={conv_os:.6f}"

    def test_bound_ordering(self):
        conv, Q_cond = self.tb.compute_converse(self.M)
        P = Q_cond / (self.tb.k_v ** self.tb.n)
        ach = self.tb.achievable_bound(self.M, P)
        assert conv <= ach + 1e-6, f"conv={conv:.4f} > ach={ach:.4f}"


class TestZChannel_n1:
    """Z-channel(0.2), non-uniform source, n=1, k_v=2."""

    def setup_method(self):
        self.P_V = np.array([0.3, 0.7])
        self.W   = _make_z_channel(0.2)
        self.M   = 2.0
        self.tb  = TypeBasedJSCC(self.P_V, self.W, n=1)
        self.os  = OneShotJSCC(self.P_V, self.W)
        _, self.Q_opt = self.os.compute_converse(self.M)
        self.P_T_VX   = self.tb.q_xgv_to_type_prior(self.Q_opt)

    def test_f_curve_matches_one_shot(self):
        k_tb, v_tb = self.tb.compute_f_curve(self.P_T_VX)
        k_os, v_os = self.os.compute_f_curve(self.Q_opt)
        _compare_curves(k_tb, v_tb, k_os, v_os, label="Z-ch n=1")

    def test_achievable_matches_one_shot(self):
        ach_tb = self.tb.achievable_bound(self.M, self.P_T_VX)
        ach_os = self.os.achievable_bound(self.M, self.Q_opt)
        assert abs(ach_tb - ach_os) < 1e-4, \
            f"type={ach_tb:.6f}  oneshot={ach_os:.6f}"

    def test_converse_matches_one_shot(self):
        conv_tb, _ = self.tb.compute_converse(self.M)
        conv_os, _ = self.os.compute_converse(self.M)
        assert abs(conv_tb - conv_os) < 1e-4, \
            f"type={conv_tb:.6f}  oneshot={conv_os:.6f}"


class TestRandom3x3_n1:
    """3×3 random channel, uniform source, n=1, k_v=3."""

    def setup_method(self):
        rng = np.random.default_rng(7)
        raw = rng.random((3, 3)) + 4 * np.eye(3)
        W   = raw / raw.sum(axis=1, keepdims=True)
        self.P_V = np.ones(3) / 3
        self.W   = W
        self.M   = 3.0
        self.tb  = TypeBasedJSCC(self.P_V, self.W, n=1)
        self.os  = OneShotJSCC(self.P_V, self.W)
        _, self.Q_opt = self.os.compute_converse(self.M)
        self.P_T_VX   = self.tb.q_xgv_to_type_prior(self.Q_opt)

    def test_f_curve_matches_one_shot(self):
        k_tb, v_tb = self.tb.compute_f_curve(self.P_T_VX)
        k_os, v_os = self.os.compute_f_curve(self.Q_opt)
        _compare_curves(k_tb, v_tb, k_os, v_os, label="3x3 n=1")

    def test_achievable_matches_one_shot(self):
        ach_tb = self.tb.achievable_bound(self.M, self.P_T_VX)
        ach_os = self.os.achievable_bound(self.M, self.Q_opt)
        assert abs(ach_tb - ach_os) < 1e-4, \
            f"type={ach_tb:.6f}  oneshot={ach_os:.6f}"

    def test_converse_matches_one_shot(self):
        conv_tb, _ = self.tb.compute_converse(self.M)
        conv_os, _ = self.os.compute_converse(self.M)
        assert abs(conv_tb - conv_os) < 1e-4, \
            f"type={conv_tb:.6f}  oneshot={conv_os:.6f}"


# ── n > 1 tests ───────────────────────────────────────────────────────────────

class TestBSC_n2:
    """
    BSC(0.1), uniform source, n=2.

    One-shot reference: OneShotJSCC(P_V^{⊗2}, W^{⊗2}) with M=4.
    Type-based: TypeBasedJSCC(P_V, W, n=2) with M=4.

    Converse should match (BSC + uniform source → optimal encoder is type-constant).
    Achievable: convert type-based Q_cond to sequence-level encoder for one-shot.
    """

    def setup_method(self):
        self.P_V  = np.array([0.5, 0.5])
        self.W    = _make_bsc(0.1)
        self.n    = 2
        self.M    = float(len(self.P_V) ** self.n)   # 4
        self.tb   = TypeBasedJSCC(self.P_V, self.W, n=self.n)
        self.os   = OneShotJSCC(_product_source(self.P_V, self.n),
                                _kronecker_power(self.W, self.n))
        # Type-based LP converse + correct prior
        self.conv_tb, self.Q_cond = self.tb.compute_converse(self.M)
        self.P_tb = self.Q_cond / (self.tb.k_v ** self.n)

    def test_f_curve_valid(self):
        k, v = self.tb.compute_f_curve(self.P_tb)
        _check_f_curve(k, v, label="BSC n=2")

    def test_converse_matches_one_shot(self):
        conv_os, _ = self.os.compute_converse(self.M)
        assert abs(self.conv_tb - conv_os) < 1e-3, \
            f"type={self.conv_tb:.6f}  oneshot={conv_os:.6f}"

    def test_achievable_matches_one_shot(self):
        """Achievable bound with the LP-optimal type-constant encoder."""
        ach_tb = self.tb.achievable_bound(self.M, self.P_tb)
        # Convert type-constant Q_cond to sequence-level encoder for one-shot
        Q_seq = _type_constant_to_sequence_encoder(self.Q_cond, self.tb)
        ach_os = self.os.achievable_bound(self.M, Q_seq)
        assert abs(ach_tb - ach_os) < 1e-3, \
            f"type={ach_tb:.6f}  oneshot={ach_os:.6f}"

    def test_bound_ordering(self):
        ach = self.tb.achievable_bound(self.M, self.P_tb)
        assert self.conv_tb <= ach + 1e-6, \
            f"conv={self.conv_tb:.4f} > ach={ach:.4f}"


class TestBSC_n3:
    """
    BSC(0.1), uniform source, n=3.
    Same checks as n=2.
    """

    def setup_method(self):
        self.P_V  = np.array([0.5, 0.5])
        self.W    = _make_bsc(0.1)
        self.n    = 3
        self.M    = float(len(self.P_V) ** self.n)   # 8
        self.tb   = TypeBasedJSCC(self.P_V, self.W, n=self.n)
        self.os   = OneShotJSCC(_product_source(self.P_V, self.n),
                                _kronecker_power(self.W, self.n))
        self.conv_tb, self.Q_cond = self.tb.compute_converse(self.M)
        self.P_tb = self.Q_cond / (self.tb.k_v ** self.n)

    def test_f_curve_valid(self):
        k, v = self.tb.compute_f_curve(self.P_tb)
        _check_f_curve(k, v, label="BSC n=3")

    def test_converse_matches_one_shot(self):
        conv_os, _ = self.os.compute_converse(self.M)
        assert abs(self.conv_tb - conv_os) < 1e-3, \
            f"type={self.conv_tb:.6f}  oneshot={conv_os:.6f}"

    def test_achievable_matches_one_shot(self):
        ach_tb = self.tb.achievable_bound(self.M, self.P_tb)
        Q_seq  = _type_constant_to_sequence_encoder(self.Q_cond, self.tb)
        ach_os = self.os.achievable_bound(self.M, Q_seq)
        assert abs(ach_tb - ach_os) < 1e-3, \
            f"type={ach_tb:.6f}  oneshot={ach_os:.6f}"

    def test_bound_ordering(self):
        ach = self.tb.achievable_bound(self.M, self.P_tb)
        assert self.conv_tb <= ach + 1e-6, \
            f"conv={self.conv_tb:.4f} > ach={ach:.4f}"


# ── memoryless prior tests ─────────────────────────────────────────────────────

def _memoryless_seq_encoder(Q_XgV, n):
    """
    Build the (k_v^n, k_x^n) sequence-level encoder for the memoryless
    encoder Q_{X|V}^n.  Q_seq[v_idx, x_idx] = prod_i Q(x_i | v_i).
    """
    k_v, k_x = Q_XgV.shape
    kv_n, kx_n = k_v ** n, k_x ** n
    Q_seq = np.ones((kv_n, kx_n))
    for pos in range(n):
        for v_idx in range(kv_n):
            v_sym = (v_idx // (k_v ** pos)) % k_v
            for x_idx in range(kx_n):
                x_sym = (x_idx // (k_x ** pos)) % k_x
                Q_seq[v_idx, x_idx] *= Q_XgV[v_sym, x_sym]
    return Q_seq


class TestMemorylessPrior:
    """
    Validate memoryless_prior(Q) end-to-end:

    Chain:
      1. n=1 : memoryless_prior(Q) == q_xgv_to_type_prior(Q)
      2. n=1 : achievable with memoryless_prior matches OneShotJSCC achievable
      3. n=2 : achievable with memoryless_prior matches OneShotJSCC (product)
               with product memoryless encoder Q_seq = prod Q(x_i|v_i)
      4. n=3 : same as n=2
      5. conv <= achievable_with_memoryless_prior for n=1,2,3

    Channels: BSC(0.1) + Bernoulli(0.1)   (non-uniform source to stress-test)
    """

    def setup_method(self):
        self.P_V = np.array([0.9, 0.1])
        self.W   = _make_bsc(0.1)
        # use a non-trivial Q (not the LP-optimal uniform)
        self.Q   = np.array([[0.7, 0.3],
                             [0.4, 0.6]])

    # ── n=1 ───────────────────────────────────────────────────────────────────

    def test_n1_memoryless_prior_matches_q_xgv(self):
        """memoryless_prior and q_xgv_to_type_prior must agree at n=1."""
        tb = TypeBasedJSCC(self.P_V, self.W, n=1)
        P_ml  = tb.memoryless_prior(self.Q)
        P_ref = tb.q_xgv_to_type_prior(self.Q)
        assert np.allclose(P_ml, P_ref, atol=1e-10), \
            f"max diff = {np.max(np.abs(P_ml - P_ref)):.2e}"

    def test_n1_achievable_matches_one_shot(self):
        """achievable_bound with memoryless_prior at n=1 == OneShotJSCC."""
        tb = TypeBasedJSCC(self.P_V, self.W, n=1)
        os = OneShotJSCC(self.P_V, self.W)
        ach_tb = tb.achievable_bound(2.0, tb.memoryless_prior(self.Q))
        ach_os = os.achievable_bound(2.0, self.Q)
        assert abs(ach_tb - ach_os) < 1e-4, \
            f"type={ach_tb:.6f}  oneshot={ach_os:.6f}"

    def test_n1_bound_ordering(self):
        tb = TypeBasedJSCC(self.P_V, self.W, n=1)
        conv, _ = tb.compute_converse(2.0)
        ach = tb.achievable_bound(2.0, tb.memoryless_prior(self.Q))
        assert conv <= ach + 1e-6, f"n=1  conv={conv:.6f} > ach={ach:.6f}"

    # ── n=2 ───────────────────────────────────────────────────────────────────

    def test_n2_achievable_matches_one_shot(self):
        """achievable_bound with memoryless_prior at n=2 == OneShotJSCC product."""
        n = 2
        tb = TypeBasedJSCC(self.P_V, self.W, n=n)
        os = OneShotJSCC(_product_source(self.P_V, n),
                         _kronecker_power(self.W, n))
        Q_seq  = _memoryless_seq_encoder(self.Q, n)
        ach_tb = tb.achievable_bound(4.0, tb.memoryless_prior(self.Q))
        ach_os = os.achievable_bound(4.0, Q_seq)
        assert abs(ach_tb - ach_os) < 1e-4, \
            f"type={ach_tb:.6f}  oneshot={ach_os:.6f}"

    def test_n2_bound_ordering(self):
        tb = TypeBasedJSCC(self.P_V, self.W, n=2)
        conv, _ = tb.compute_converse(4.0)
        ach = tb.achievable_bound(4.0, tb.memoryless_prior(self.Q))
        assert conv <= ach + 1e-6, f"n=2  conv={conv:.6f} > ach={ach:.6f}"

    # ── n=3 ───────────────────────────────────────────────────────────────────

    def test_n3_achievable_matches_one_shot(self):
        """achievable_bound with memoryless_prior at n=3 == OneShotJSCC product."""
        n = 3
        tb = TypeBasedJSCC(self.P_V, self.W, n=n)
        os = OneShotJSCC(_product_source(self.P_V, n),
                         _kronecker_power(self.W, n))
        Q_seq  = _memoryless_seq_encoder(self.Q, n)
        ach_tb = tb.achievable_bound(8.0, tb.memoryless_prior(self.Q))
        ach_os = os.achievable_bound(8.0, Q_seq)
        assert abs(ach_tb - ach_os) < 1e-4, \
            f"type={ach_tb:.6f}  oneshot={ach_os:.6f}"

    def test_n3_bound_ordering(self):
        tb = TypeBasedJSCC(self.P_V, self.W, n=3)
        conv, _ = tb.compute_converse(8.0)
        ach = tb.achievable_bound(8.0, tb.memoryless_prior(self.Q))
        assert conv <= ach + 1e-6, f"n=3  conv={conv:.6f} > ach={ach:.6f}"


# ── standalone runner ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import traceback

    test_classes = [
        TestBSC_n1,
        TestZChannel_n1,
        TestRandom3x3_n1,
        TestBSC_n2,
        TestBSC_n3,
        TestMemorylessPrior,
    ]

    passed = failed = 0
    for cls in test_classes:
        methods = sorted(m for m in dir(cls) if m.startswith('test_'))
        for name in methods:
            obj = cls()
            if hasattr(obj, 'setup_method'):
                obj.setup_method()
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}::{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {cls.__name__}::{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed + failed} tests: {passed} passed, {failed} failed.")
