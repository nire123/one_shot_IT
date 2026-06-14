"""
Channel Coding Achievable Bound - Type-Based Implementation
============================================================

Type-based implementation for memoryless channels with type-constant priors.

Given:
- Single-letter channel W_single (k_x × k_y matrix)
- Type-based prior P(T_X) over input types
- Blocklength n

Compute:
- F-curve using type enumeration (polynomial in n)
- Error probability via integration

Advantage: Scales as O(n^k) instead of O(k^n)

Author: Nir
Date: 2026-04-03
"""

import numpy as np
import cvxpy as cp

from fbl.F_curve import merge_piecewise_linear_curves, integrate_curve_channel_coding_exact, compare_F_curves

from fbl.type_class_core import (
    enumerate_type_class,
    enumerate_conditional_type_class,
    composition_to_index,
    composition_count,
    log_size_type_class,
    log_size_conditional_type_class, conditional_enum
)


from fbl.type_based_utils import type_prior_to_one_shot

# Import one-shot implementation
from fbl.one_shot_channel import _build_F_curve as build_F_curve_one_shot


from fbl.channel_achievable_utils import (
    kronecker_power,
    binary_symmetric_channel,
    z_channel, binary_erasure_channel
)

class TypeBasedChannel:
    def __init__(self, W_single, n, verbose=False):
        
        self.verbose = verbose
        # Validate inputs
        assert np.allclose(W_single.sum(axis=1), 1.0), "W_single rows must sum to 1"
                
        log_W = np.log(W_single+1e-100)
        k_x, k_y = log_W.shape
        
        cond_y_x = conditional_enum(n, k_y, k_x)    
        self.cond_y_x = cond_y_x
        
        self.alpha_coeffs = np.empty((cond_y_x.len,))
        self.R_to_Q =  np.empty((cond_y_x.len,), dtype=np.int32)
        self.R_Q_ratio =  np.empty((cond_y_x.len,))
        
        self.num_q = composition_count(n, k_x)
        self.num_R = cond_y_x.len
        
        for ix, (i_T_y, T_y, i_T_x_given_y, T_yx, log_size_conditional_type_class_T_yx) in enumerate(self.cond_y_x.enumerate()):
    
            T_x = T_yx.sum(axis=0)
            i_T_x = composition_to_index(T_x)
            
            log_T_Y = cond_y_x.size_cond_given_prime[i_T_y]
                            
            # Alpha coefficient: |T_Y| · W(y|x)        
            log_alpha = (T_yx.T*log_W).sum() + log_T_Y
            alpha = np.exp(log_alpha)
                    
            # Flat Q: Q(T_X) · |T_{X|Y}| / |T_X|
            log_size_T_YX = log_size_conditional_type_class(T_yx)
            log_size_T_X = cond_y_x.size_prime[i_T_x]
            
            ratio = np.exp(log_size_T_YX - log_size_T_X)
            
            self.alpha_coeffs[ix] = alpha
            self.R_to_Q[ix] = i_T_x
            self.R_Q_ratio[ix] = ratio
    
    def build_F_curve_type_based(self, P_T_X):
        assert np.isclose(P_T_X.sum(), 1.0), f"P_T_X sums to {P_T_X.sum()}"
        
        assert self.num_q == len(P_T_X)
        
        all_knots = []
        all_values = []
        
        P_X_times_ratio = P_T_X[self.R_to_Q]*self.R_Q_ratio
        
        
        # For each output type
        for s,e in self.cond_y_x.iterate_cond():
            
            alpha_coeffs1, P_X_times_ratio1 = self.alpha_coeffs[s:e], P_X_times_ratio[s:e]
            
            ix = np.argsort(alpha_coeffs1)[::-1]
            
            alpha_coeffs_sorted = alpha_coeffs1[ix]
            P_X_times_ratio_sorted = P_X_times_ratio1[ix]
    
            knots = np.concatenate([[0.0], np.cumsum(P_X_times_ratio_sorted)])
            values = np.concatenate([[0.0], np.cumsum(alpha_coeffs_sorted * P_X_times_ratio_sorted)])
            
            all_knots.append(knots)
            all_values.append(values)
                    
        # Merge all curves
        merged_knots, merged_F = merge_piecewise_linear_curves(all_knots, all_values)
        
        # Verify normalization
        if not np.isclose(merged_F[-1], 1.0, atol=1e-3):
            print(f"Warning: F-curve doesn't sum to 1: {merged_F[-1]}")
        
        return merged_knots, merged_F
    
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
            Error probability (1 - success_prob), or None if infeasible
        """
        constraints = []
        
        # Variables
        Q_var = cp.Variable((self.num_q,), nonneg=True)
        R_var = cp.Variable((self.num_R,), nonneg=True)
        

        # (C1) Q is a distribution over input types
        constraints.append(cp.sum(Q_var) == 1.0)
        
        # (C2) R[r] <= Q[T_X(r)] * M * ratio[r]
        constraints.append(
            R_var <= cp.multiply(Q_var[self.R_to_Q], M * self.R_Q_ratio)
        )
        
        # (C3) For each y-type block: sum_r R[r] <= 1
        # for i_T_Y, (st, ed) in self.y_blocks.items():
        for st, ed in self.cond_y_x.iterate_cond():
            constraints.append(cp.sum(R_var[st:ed]) <= 1.0)
        
        # Objective: maximize success probability
        objective = cp.Maximize(cp.sum(cp.multiply(R_var, self.alpha_coeffs / M)))
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCIPY, scipy_options={"method": "highs-ds"}, 
                     verbose=self.verbose)
        
        self.status = problem.status
        
        if self.status not in ["optimal", "optimal_inaccurate"]:
            self.Q_values = None
            self.success_prob = None
            self.error_prob = None
            return None
        
        self.Q_values = Q_var.value
        self.success_prob = float(problem.value)
        self.error_prob = 1.0 - self.success_prob
        
        return self.error_prob
        
# ============================================================================
# Type-Based F-Curve Construction
# ============================================================================

def build_F_curve_type_based(W_single, P_T_X, n):
    """
    Build F-curve using type-based computation.
    
    For memoryless channel W^⊗n and type-constant prior P(T_X),
    compute F-curve by enumerating types instead of sequences.
    
    Complexity: O(n^{k_x + k_y}) instead of O(k^n) for one-shot.
    
    Parameters
    ----------
    W_single : np.ndarray, shape (k_x, k_y)
        Single-letter channel transition matrix
    P_T_X : np.ndarray, shape (num_input_types,)
        Type-based prior over input types
        Must sum to 1
    n : int
        Blocklength
    
    Returns
    -------
    knots : np.ndarray
        F-curve knot positions
    F_vals : np.ndarray
        F-curve values
    """
    
    tb = TypeBasedChannel(W_single, n)
    return tb.build_F_curve_type_based(P_T_X)

# ============================================================================
# High-level API
# ============================================================================

def compute_achievable_theory_type_based(W_single, P_T_X, n, M, num_refined_points=1000):
    """
    Compute theoretical achievable error probability using type-based method.
    
    Parameters
    ----------
    W_single : np.ndarray, shape (k_x, k_y)
        Single-letter channel
    P_T_X : np.ndarray
        Type-based prior
    n : int
        Blocklength
    M : int
        Codebook size
    num_refined_points : int
        Grid refinement
    
    Returns
    -------
    float
        Error probability (theory)
    """
    
    log_W = np.log(W_single+1e-100)
    
    # Build F-curve
    knots, F_vals = build_F_curve_type_based(W_single, P_T_X, n)
        
    # Integrate        
    P_error = integrate_curve_channel_coding_exact(knots, F_vals, M, num_refined_points=1000)
    
    
    return P_error


class ChannelCodingAchievableTypeBased:
    """
    Type-based achievable bound wrapper.
    
    Parallel to ChannelCodingAchievable but works with types.
    
    Usage:
        W_single = np.array([[0.9, 0.1], [0.1, 0.9]])  # BSC
        P_T_X = random_type_prior(n=5, k_x=2)
        
        cc = ChannelCodingAchievableTypeBased(W_single, P_T_X, n=5)
        P_error = cc.theory(M=4)
    """
    
    def __init__(self, W_single, P_T_X, n):
        """
        Initialize type-based achievable bound.
        
        Parameters
        ----------
        W_single : np.ndarray, shape (k_x, k_y)
            Single-letter channel
        P_T_X : np.ndarray
            Type-based prior
        n : int
            Blocklength
        """
        self.W_single = np.array(W_single)
        self.P_T_X = np.array(P_T_X)
        self.n = n
        
        self.k_x, self.k_y = W_single.shape
        
        # Validate
        num_types = composition_count(n, self.k_x)
        assert P_T_X.shape == (num_types,), \
            f"P_T_X shape {P_T_X.shape} != ({num_types},)"
        assert np.isclose(P_T_X.sum(), 1.0), f"P_T_X sums to {P_T_X.sum()}"
        assert np.allclose(W_single.sum(axis=1), 1.0), "W_single rows must sum to 1"
    
    def theory(self, M, num_refined_points=1000):
        """Compute theoretical achievable error probability."""
        return compute_achievable_theory_type_based(
            self.W_single, self.P_T_X, self.n, M, num_refined_points
        )
    
    def build_curve(self):
        """Build and return F-curve (for debugging/inspection)."""
        return build_F_curve_type_based(self.W_single, self.P_T_X, self.n)



def test_type_based_f_curve_single_ch(W_single, n):
    k_x = W_single.shape[0]
    
    # Uniform type prior
    num_types = composition_count(n, k_x)
    P_T_X = np.ones(num_types) / num_types
        
    # Build F-curve
    knots, F_vals = build_F_curve_type_based(W_single, P_T_X, n)
        
    W_n = kronecker_power(W_single, n)
    Q_one_shot = type_prior_to_one_shot(P_T_X, n, k_x)
    knots_one, F_one = build_F_curve_one_shot(W_n, Q_one_shot)
    
    res = compare_F_curves(knots, F_vals, knots_one, F_one)
    
    return res['all_close']

# ============================================================================
# Testing
# ============================================================================

def test_type_based_f_curve():
    """Basic test of type-based F-curve construction."""
    print("="*80)
    print("Testing Type-Based F-Curve Construction")
    print("="*80)
    
    
    for ch_f in [binary_erasure_channel, binary_symmetric_channel, z_channel]:
        W_single = ch_f(0.1)
        complexity = np.prod(np.array(W_single.shape)-1)
        N = 10 if complexity == 1 else 7
        for n in range(2,N):
            res = test_type_based_f_curve_single_ch(W_single, n)
            pass_fail = 'Pass' if res else 'Fail'
            print(f'Test channel: {ch_f.__name__}; n={n} : {pass_fail}')
            assert res
            
        
    # Simple BSC
    W_single = np.array([[0.9, 0.1],
                         [0.1, 0.9]])
        
    n = 3
    k_x = W_single.shape[0]
    
    # Uniform type prior
    num_types = composition_count(n, k_x)
    P_T_X = np.ones(num_types) / num_types
    
    print(f"\nChannel: BSC(0.1)")
    print(f"Blocklength: n={n}")
    print(f"Uniform type prior ({num_types} types)")
    
    # Build F-curve
    print("\nBuilding F-curve...")
    knots, F_vals = build_F_curve_type_based(W_single, P_T_X, n)
    
    
    W_n = kronecker_power(W_single, n)
    Q_one_shot = type_prior_to_one_shot(P_T_X, n, k_x)
    knots_one, F_one = build_F_curve_one_shot(W_n, Q_one_shot)
    
    
    res = compare_F_curves(knots, F_vals, knots_one, F_one)
    
    assert res['all_close']

    print(f"F-curve has {len(knots)} knots")
    print(f"F-curve range: [{F_vals.min():.6f}, {F_vals.max():.6f}]")
    print(f"F-curve normalized: {np.isclose(F_vals[-1], 1.0)}")
    
    # Compute error for different M
    print("\nError probabilities:")
    for M in [2, 3, 4, 5]:
        
        # Integrate            
        P_error = integrate_curve_channel_coding_exact(knots, F_vals, M, num_refined_points=1000)
        
        
        print(f"  M={M}: P_error = {P_error:.6f}")
    
    print("\n" + "="*80)
    print("✓ Type-based F-curve construction successful")
    print("="*80)


if __name__ == '__main__':
    test_type_based_f_curve()
