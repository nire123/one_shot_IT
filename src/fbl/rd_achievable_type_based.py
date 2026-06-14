"""
Rate-Distortion Achievable Bound - Type-Based Implementation
=============================================================

Type-based implementation for memoryless sources with type-constant priors.

Given:
- Single-letter source P_X_single over alphabet X
- Single-letter distortion d_single[x,y]
- Type-based prior P(T_Y) over reconstruction types
- Blocklength n

Compute:
- A-curve using type enumeration (polynomial in n)
- Expected distortion via integration

Advantage: Scales as O(n^k) instead of O(k^n)

Author: Nir
Date: 2026-04-03
"""

import numpy as np
import cvxpy as cp

from fbl.type_class_core import (
    enumerate_type_class,
    enumerate_conditional_type_class,
    composition_to_index,
    composition_count,
    log_size_type_class,
    log_size_conditional_type_class, 
    conditional_enum
)
from fbl.F_curve import merge_piecewise_linear_curves, integrate_curve_rd_exact, compare_F_curves


from fbl.type_based_utils import (
    random_type_prior,
)

from fbl.achievable_utils import (
    binary_memoryless_source,
    hamming_distortion
)

from fbl.type_based_utils import type_prior_to_one_shot

from fbl.one_shot_rd import _build_A_curve as build_A_curve_one_shot

# ============================================================================
# Type-Based A-Curve Construction
# ============================================================================


class TypeBasedRateDistortion:
    def __init__(self, P_X_single, d_single, n, verbose=False):
        
        self.verbose = verbose
        # Validate inputs
        assert np.allclose(P_X_single.sum(), 1.0), "W_single rows must sum to 1"
                
        k_x, k_y = d_single.shape
        
        cond_x_y = conditional_enum(n, k_x, k_y)    
        self.cond_x_y = cond_x_y
        
        self.d_coeffs = np.empty((cond_x_y.len,))
        # self.d_coeffs1 = np.empty((cond_x_y.len,))
        self.R_to_Q =  np.empty((cond_x_y.len,), dtype=np.int32)
        self.R_Q_ratio =  np.empty((cond_x_y.len,))
        
        self.num_q = composition_count(n, k_y)
        self.num_R = cond_x_y.len


        # Source probability for this type
        
        self.P_X_prob = np.empty(composition_count(n, k_x))
        
        for i_T_x, T_X in enumerate(enumerate_type_class(n, k_x)):
            
            log_T_X = log_size_type_class(T_X)
            log_prob = np.sum(T_X * np.log(P_X_single + 1e-100))
            P_X_type = np.exp(log_T_X + log_prob)
            self.P_X_prob[i_T_x] = P_X_type
            
        for ix, (i_T_x, T_x, i_T_y_given_x, T_xy, log_size_conditional_type_class_T_xy) in enumerate(self.cond_x_y.enumerate()):
    
            T_y = T_xy.sum(axis=0)
            i_T_y = composition_to_index(T_y)
            
            log_T_X = cond_x_y.size_cond_given_prime[i_T_x]
                                        
            total_d = (T_xy*d_single).sum()
                                
            # Flat Q: Q(T_Y) · |T_{Y|X}| / |T_T|
            log_size_T_XY = log_size_conditional_type_class(T_xy)
            log_size_T_Y = cond_x_y.size_prime[i_T_y]
            
            ratio = np.exp(log_size_T_XY - log_size_T_Y)
                        
            self.d_coeffs[ix] = total_d*self.P_X_prob[i_T_x]
            self.R_to_Q[ix] = i_T_y
            self.R_Q_ratio[ix] = ratio
    
    def build_A_curve_type_based(self, P_T_Y):
        assert np.isclose(P_T_Y.sum(), 1.0), f"P_T_Y sums to {P_T_Y.sum()}"
        
        assert self.num_q == len(P_T_Y)
                
        all_knots = []
        all_values = []
                    
        P_Y_times_ratio = P_T_Y[self.R_to_Q]*self.R_Q_ratio
                    
        # For each source type (mirrors: for each source sequence in one-shot)
        for ix,(st,ed)  in enumerate(self.cond_x_y.iterate_cond()):        
            d_coeffs, P_Y_times_ratio1 = self.d_coeffs[st:ed], P_Y_times_ratio[st:ed]            
            sort_ix = np.argsort(d_coeffs)
            
            d_coeffs_sorted = d_coeffs[sort_ix]
            P_Y_times_ratio_sorted = P_Y_times_ratio1[sort_ix]

            knots = np.concatenate([[0.0], np.cumsum(P_Y_times_ratio_sorted)])
            values = np.concatenate([[0.0], np.cumsum(d_coeffs_sorted * P_Y_times_ratio_sorted )])
                
            all_knots.append(knots)
            all_values.append(values)
        
        # Merge all curves
        merged_knots, merged_A = merge_piecewise_linear_curves(all_knots, all_values)
        
        return merged_knots, merged_A
    
    def compute_opt_Q(self, M):
        """
        Solve type-based LP to find optimal type prior.
        
        Parameters
        ----------
        M : float
            Codebook size (real-valued allowed)
        
        Returns
        -------
        float or None
            Expected distortion, or None if infeasible
        """
        constraints = []
        
        # Variables
        Q_var = cp.Variable((self.num_q,), nonneg=True)
        R_var = cp.Variable((self.num_R,), nonneg=True)
        
        # (C1) Q is a distribution over reconstruction types
        constraints.append(cp.sum(Q_var) == 1.0)
        
        # (C2) R[r] <= Q[T_Y(r)] * M * ratio[r]
        constraints.append(
            R_var <= cp.multiply(Q_var[self.R_to_Q], M * self.R_Q_ratio)
        )
        
        # (C3) For each x-type block: sum_r R[r] <= 1
        # for i_T_X, (st, ed) in self.x_blocks.items():
        for ix,(st,ed)  in enumerate(self.cond_x_y.iterate_cond()):
            
            constraints.append(cp.sum(R_var[st:ed]) >= 1.0)
        
        # Objective: minimize expected distortion
        objective = cp.Minimize(cp.sum(cp.multiply(R_var, self.d_coeffs / M)))
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.SCIPY, scipy_options={"method": "highs-ds"},
                         verbose=self.verbose)
        except Exception:
            problem.solve(solver=cp.CLARABEL, verbose=self.verbose)
        
        self.status = problem.status
        
        if self.status not in ["optimal", "optimal_inaccurate"]:
            self.Q_values = None
            self.distortion = None
            return None
        
        self.Q_values = Q_var.value
        self.distortion = float(problem.value)

        return self.distortion




def build_A_curve_type_based(P_X_single, P_T_Y, d_single, n):
    """
    Build A-curve using type-based computation.
    
    Mirrors one-shot structure:
    - One-shot: iterate over source sequences
    - Type-based: iterate over source types
    
    For memoryless source and type-constant reconstruction prior,
    compute A-curve by enumerating types instead of sequences.
    
    Complexity: O(n^{k_x + k_y}) instead of O(k^n) for one-shot.
    
    Parameters
    ----------
    P_X_single : np.ndarray, shape (k_x,)
        Single-letter source distribution
    P_T_Y : np.ndarray, shape (num_recon_types,)
        Type-based prior over reconstruction types
        Must sum to 1
    d_single : np.ndarray, shape (k_x, k_y)
        Single-letter distortion matrix
    n : int
        Blocklength
    
    Returns
    -------
    knots : np.ndarray
        A-curve knot positions
    A_vals : np.ndarray
        A-curve values
    """
    
    # Validate inputs
    assert np.isclose(P_T_Y.sum(), 1.0), f"P_T_Y sums to {P_T_Y.sum()}"
    assert np.isclose(P_X_single.sum(), 1.0), f"P_X_single sums to {P_X_single.sum()}"
        
    tb = TypeBasedRateDistortion(P_X_single, d_single, n)
    return tb.build_A_curve_type_based(P_T_Y)
    


# ============================================================================
# High-level API
# ============================================================================

def compute_achievable_theory_type_based(P_X_single, P_T_Y, d_single, n, M, 
                                          num_refined_points=1000):
    """
    Compute theoretical achievable distortion using type-based method.
    
    Parameters
    ----------
    P_X_single : np.ndarray, shape (k_x,)
        Single-letter source distribution
    P_T_Y : np.ndarray
        Type-based reconstruction prior
    d_single : np.ndarray, shape (k_x, k_y)
        Single-letter distortion matrix
    n : int
        Blocklength
    M : int
        Codebook size
    num_refined_points : int
        Grid refinement
    
    Returns
    -------
    float
        Expected distortion (theory)
    """
    # Build A-curve
    knots, A_vals = build_A_curve_type_based(P_X_single, P_T_Y, d_single, n)
    
    # Integrate    
    D_rand = integrate_curve_rd_exact(knots, A_vals, M, num_refined_points=1000)

    
    return D_rand


class RateDistortionAchievableTypeBased:
    """
    Type-based achievable bound wrapper for rate-distortion.
    
    Parallel to channel coding type-based implementation.
    
    Usage:
        P_X_single = np.array([0.9, 0.1])  # BMS
        d_single = np.array([[0, 1], [1, 0]])  # Hamming
        P_T_Y = uniform_type_prior(n=5, k_y=2)
        
        rd = RateDistortionAchievableTypeBased(P_X_single, P_T_Y, d_single, n=5)
        D = rd.theory(M=5)
    """
    
    def __init__(self, P_X_single, P_T_Y, d_single, n):
        """
        Initialize type-based achievable bound.
        
        Parameters
        ----------
        P_X_single : np.ndarray, shape (k_x,)
            Single-letter source distribution
        P_T_Y : np.ndarray
            Type-based reconstruction prior
        d_single : np.ndarray, shape (k_x, k_y)
            Single-letter distortion matrix
        n : int
            Blocklength
        """
        self.P_X_single = np.array(P_X_single)
        self.P_T_Y = np.array(P_T_Y)
        self.d_single = np.array(d_single)
        self.n = n
        
        self.k_x = len(P_X_single)
        self.k_y = d_single.shape[1]
        
        # Validate
        num_types = composition_count(n, self.k_y)
        assert P_T_Y.shape == (num_types,), \
            f"P_T_Y shape {P_T_Y.shape} != ({num_types},)"
        assert np.isclose(P_T_Y.sum(), 1.0), f"P_T_Y sums to {P_T_Y.sum()}"
        assert np.isclose(P_X_single.sum(), 1.0), f"P_X_single sums to {P_X_single.sum()}"
        assert d_single.shape == (self.k_x, self.k_y), \
            f"d_single shape {d_single.shape} != ({self.k_x}, {self.k_y})"
    
    def theory(self, M, num_refined_points=1000):
        """Compute theoretical achievable distortion."""
        return compute_achievable_theory_type_based(
            self.P_X_single, self.P_T_Y, self.d_single, self.n, M, num_refined_points
        )
    
    def build_curve(self):
        """Build and return A-curve (for debugging/inspection)."""
        return build_A_curve_type_based(self.P_X_single, self.P_T_Y, self.d_single, self.n)


# ============================================================================
# Testing
# ============================================================================
def compare_type_based(P_X_single, d_single, n):
        
    # Random type prior
    
    P_T_Y = random_type_prior(n, d_single.shape[1], seed=42)
            
    # Type-based computation
    knots_type, A_type = build_A_curve_type_based(P_X_single, P_T_Y, d_single, n)
    
    # One-shot computation
    
    P_X = P_X_single
    d = d_single
    for _ in range(n-1):
        P_X = np.kron(P_X, P_X_single)
        
        # d = d[:, :, None, None] + d_single[None, None, :, :]
        d = d[:, None, :, None] + d_single[None, :, None, :]
        # Reshape to flatten the Kronecker-like structure
        shape = d.shape
        d = d.reshape(shape[0] * shape[1], shape[2] * shape[3])

    Q_Y = type_prior_to_one_shot(P_T_Y, n, d_single.shape[1])
    knots_one, A_one = build_A_curve_one_shot(P_X, Q_Y, d)
        
    return compare_F_curves(knots_type, A_type, knots_one, A_one)
    




def test_random_type_prior_bms():
    """
    Test 3: Random type-constant prior on BMS.
    """
    print("\n" + "="*80)
    print("TEST 3: Random Type-Constant Prior on BMS")
    print("="*80)
    
    p = 0.15
    n = 4
    
    P_X_single = np.array([1-p, p])
    d_single = np.array([[0.0, 1.0],
                         [1.0, 0.0]])
    
    # Random type prior
    P_T_Y = random_type_prior(n, 2, seed=42)
    
    print(f"\nSetup:")
    print(f"  Source: BMS({p})")
    print(f"  Distortion: Hamming")
    print(f"  Blocklength: n={n}")
    print(f"  Reconstruction prior: Random type-constant (seed=42)")
    
    print(f"\nType prior (first 5 types):")
    from fbl.type_class_core import enumerate_type_class
    for i, T in enumerate(enumerate_type_class(n, 2)):
        if i >= 5:
            break
        print(f"  T={T}: P(T)={P_T_Y[i]:.6f}")
    
    # Type-based computation
    knots_type, A_type = build_A_curve_type_based(P_X_single, P_T_Y, d_single, n)
    
    P_X = P_X_single
    d = d_single
    for _ in range(n-1):
        P_X = np.kron(P_X, P_X_single)
        
        # d = d[:, :, None, None] + d_single[None, None, :, :]
        d = d[:, None, :, None] + d_single[None, :, None, :]
        # Reshape to flatten the Kronecker-like structure
        shape = d.shape
        d = d.reshape(shape[0] * shape[1], shape[2] * shape[3])
        
    d1 = d
    # One-shot computation
    d = hamming_distortion(n)
    Q_Y = type_prior_to_one_shot(P_T_Y, n, 2)
    knots_one, A_one = build_A_curve_one_shot(P_X, Q_Y, d)
    
    # Compare curves
    print(f"\nA-curve comparison:")
    print(f"  Type-based: {len(knots_type)} knots, range [{A_type.min():.6f}, {A_type.max():.6f}]")
    print(f"  One-shot:   {len(knots_one)} knots, range [{A_one.min():.6f}, {A_one.max():.6f}]")
    
    comparison = compare_F_curves(knots_type, A_type, knots_one, A_one)
    
    assert comparison['all_close']


def test_type_based_a_curve():
    """Basic test of type-based A-curve construction."""
    print("="*80)
    print("Testing Type-Based A-Curve Construction")
    print("="*80)
    
    # Binary source, Hamming distortion
    P_X_single = np.array([0.9, 0.1])  # BMS(0.1)
    d_single = np.array([[0.0, 1.0],
                         [1.0, 0.0]])
    
    n = 3
    k_y = 2
    
    # Uniform type prior
    num_types = composition_count(n, k_y)
    P_T_Y = np.ones(num_types) / num_types
    
    print(f"\nSource: BMS(0.1)")
    print(f"Distortion: Hamming")
    print(f"Blocklength: n={n}")
    print(f"Uniform type prior ({num_types} types)")
    
    # Build A-curve
    print("\nBuilding A-curve...")
    knots, A_vals = build_A_curve_type_based(P_X_single, P_T_Y, d_single, n)
    
    print(f"A-curve has {len(knots)} knots")
    print(f"A-curve range: [{A_vals.min():.6f}, {A_vals.max():.6f}]")
    
    # Compute distortion for different M
    print("\nExpected distortions:")
    for M in [2, 3, 4, 5]:        
        # Integrate            
        D = integrate_curve_rd_exact(knots, A_vals, M, num_refined_points=1000)
        
    
        print(f"  M={M}: D = {D:.6f}")
    
    print("\n" + "="*80)
    print("✓ Type-based A-curve construction successful")
    print("="*80)


if __name__ == '__main__':
    
    
    
    for k_x, k_y in [[2,2], [2,3], [3,2], [3,3], [3,4], [4,3]]:
        for n in range(1,5):
            P_X_single = np.random.random(k_x)
            P_X_single = P_X_single/P_X_single.sum()
            
            d_single = np.random.random((k_x, k_y))
            
            r = compare_type_based(P_X_single, d_single, n)
            
            pas_fail = 'Pass' if r['all_close'] else 'Fail'
            print(f"Test: k_x = {k_x}; k_y = {k_y}; n = {n}: {pas_fail}")
            assert r['all_close']
    
    
    test_random_type_prior_bms()
    test_type_based_a_curve()
