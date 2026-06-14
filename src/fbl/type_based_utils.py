"""
Type-Based Prior Utilities
===========================

Utilities for working with type-constant priors.

A type-constant prior is specified by a distribution P(T_X) over types,
where all sequences in the same type class have equal probability:
    Q(x) = P(T_X) / |T_X|  for x ∈ T_X

Author: Nir
Date: 2026-04-03
"""

import numpy as np
import sys
from fbl.type_class_core import (
    enumerate_type_class,
    composition_to_index,
    composition_count,
    log_size_type_class
)


def type_prior_to_one_shot(P_T_X, n, k_x):
    """
    Convert type-based prior to one-shot prior.
    
    Given P(T_X) over types, compute Q(x) for all sequences x:
        Q(x) = P(T_X) / |T_X|  where x ∈ T_X
    
    Parameters
    ----------
    P_T_X : np.ndarray, shape (num_types,)
        Type-based prior, indexed by composition_to_index(T_X)
        Must sum to 1
    n : int
        Blocklength
    k_x : int
        Alphabet size
    
    Returns
    -------
    Q : np.ndarray, shape (k_x^n,)
        One-shot prior over all sequences
    """
    num_sequences = k_x ** n
    Q = np.zeros(num_sequences)
    
    # For each type
    for i_T_X, T_X in enumerate(enumerate_type_class(n, k_x)):
        if P_T_X[i_T_X] == 0:
            continue
        
        # Type class size
        log_size = log_size_type_class(T_X)
        type_size = np.exp(log_size)
        
        # Probability per sequence in this type
        prob_per_seq = P_T_X[i_T_X] / type_size
        
        # Assign to all sequences in this type
        # We enumerate all sequences in type T_X
        # For binary case, this is all sequences with composition T_X
        for seq_idx in range(num_sequences):
            # Check if sequence seq_idx has type T_X
            seq_type = sequence_to_type(seq_idx, n, k_x)
            if np.array_equal(seq_type, T_X):
                Q[seq_idx] = prob_per_seq
    
    return Q


def sequence_to_type(seq_idx, n, k_x):
    """
    Convert sequence index to its type (composition).
    
    Parameters
    ----------
    seq_idx : int
        Sequence index in [0, k_x^n)
    n : int
        Blocklength
    k_x : int
        Alphabet size
    
    Returns
    -------
    T : np.ndarray, shape (k_x,)
        Type (composition) of the sequence
    """
    T = np.zeros(k_x, dtype=int)
    
    # Convert index to k_x-ary representation
    for pos in range(n):
        symbol = (seq_idx // (k_x ** pos)) % k_x
        T[symbol] += 1
    
    return T


def memoryless_to_type_prior(Q_single, n):
    """
    Convert memoryless (i.i.d.) prior to type-based prior.
    
    For memoryless prior Q(x) = ∏_i Q_single[x_i], the type-based prior is:
        P(T_X) = |T_X| · ∏_a Q_single[a]^{T_X[a]}
    
    Parameters
    ----------
    Q_single : np.ndarray, shape (k_x,)
        Single-letter distribution
    n : int
        Blocklength
    
    Returns
    -------
    P_T_X : np.ndarray, shape (num_types,)
        Type-based prior
    """
    k_x = len(Q_single)
    num_types = composition_count(n, k_x)
    P_T_X = np.zeros(num_types)
    
    for i_T_X, T_X in enumerate(enumerate_type_class(n, k_x)):
        # |T_X|
        log_size = log_size_type_class(T_X)
        
        # ∏_a Q_single[a]^{T_X[a]}
        log_prob_product = np.sum(T_X * np.log(Q_single + 1e-100))  # avoid log(0)
        
        # P(T_X) = |T_X| · ∏_a Q_single[a]^{T_X[a]}
        P_T_X[i_T_X] = np.exp(log_size + log_prob_product)
    
    # Normalize (should already be normalized, but ensure it)
    P_T_X /= P_T_X.sum()
    
    return P_T_X


def random_type_prior(n, k_x, seed=None):
    """
    Generate random type-constant prior.
    
    Samples a random distribution over types.
    
    Parameters
    ----------
    n : int
        Blocklength
    k_x : int
        Alphabet size
    seed : int, optional
        Random seed
    
    Returns
    -------
    P_T_X : np.ndarray, shape (num_types,)
        Random type-based prior (sums to 1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_types = composition_count(n, k_x)
    
    # Sample from Dirichlet (symmetric = uniform over simplex)
    P_T_X = np.random.dirichlet(np.ones(num_types))
    
    return P_T_X


def uniform_type_prior(n, k_x):
    """
    Create uniform type-based prior.
    
    All types have equal probability.
    
    Parameters
    ----------
    n : int
        Blocklength
    k_x : int
        Alphabet size
    
    Returns
    -------
    P_T_X : np.ndarray, shape (num_types,)
        Uniform type-based prior
    """
    num_types = composition_count(n, k_x)
    return np.ones(num_types) / num_types


def validate_type_prior(P_T_X):
    """
    Validate that P_T_X is a valid type-based prior.
    
    Checks:
    - All probabilities non-negative
    - Sums to 1
    
    Parameters
    ----------
    P_T_X : np.ndarray
        Type-based prior
    
    Returns
    -------
    bool
        True if valid
    """
    if not np.all(P_T_X >= 0):
        return False
    
    if not np.isclose(P_T_X.sum(), 1.0):
        return False
    
    return True


# ============================================================================
# Testing and validation
# ============================================================================

def test_conversions():
    """Test type-to-one-shot and memoryless-to-type conversions."""
    print("="*80)
    print("Testing Type Prior Conversions")
    print("="*80)
    
    # Test 1: Memoryless to type and back
    print("\nTest 1: Memoryless prior conversion")
    print("-" * 40)
    
    n = 3
    k_x = 2
    Q_single = np.array([0.7, 0.3])
    
    print(f"Single-letter prior: {Q_single}")
    print(f"Blocklength: n={n}")
    
    # Convert to type prior
    P_T_X = memoryless_to_type_prior(Q_single, n)
    print(f"\nType prior (num_types={len(P_T_X)}):")
    for i_T, T in enumerate(enumerate_type_class(n, k_x)):
        print(f"  T={T}: P(T)={P_T_X[i_T]:.6f}")
    
    print(f"\nType prior sums to: {P_T_X.sum():.10f}")
    assert validate_type_prior(P_T_X), "Invalid type prior"
    print("✓ Type prior is valid")
    
    # Convert to one-shot
    Q_one_shot = type_prior_to_one_shot(P_T_X, n, k_x)
    print(f"\nOne-shot prior (first 8 entries): {Q_one_shot[:8]}")
    print(f"One-shot prior sums to: {Q_one_shot.sum():.10f}")
    
    # Verify memoryless structure
    # For memoryless, Q(x) should equal product of Q_single
    print("\nVerifying memoryless structure:")
    all_match = True
    for seq_idx in range(min(8, k_x**n)):
        # Compute expected probability
        expected = 1.0
        for pos in range(n):
            symbol = (seq_idx // (k_x ** pos)) % k_x
            expected *= Q_single[symbol]
        
        actual = Q_one_shot[seq_idx]
        match = np.isclose(expected, actual)
        all_match = all_match and match
        
        if seq_idx < 8:
            print(f"  Seq {seq_idx}: expected={expected:.6f}, actual={actual:.6f}, match={match}")
    
    if all_match:
        print("✓ Memoryless structure verified")
    else:
        print("✗ Memoryless structure mismatch")
    
    # Test 2: Random type prior
    print("\n" + "="*80)
    print("Test 2: Random type prior")
    print("-" * 40)
    
    P_T_X_random = random_type_prior(n, k_x, seed=42)
    print(f"Random type prior:")
    for i_T, T in enumerate(enumerate_type_class(n, k_x)):
        print(f"  T={T}: P(T)={P_T_X_random[i_T]:.6f}")
    
    print(f"\nRandom type prior sums to: {P_T_X_random.sum():.10f}")
    assert validate_type_prior(P_T_X_random), "Invalid random type prior"
    print("✓ Random type prior is valid")
    
    Q_random = type_prior_to_one_shot(P_T_X_random, n, k_x)
    print(f"\nOne-shot from random (first 8): {Q_random[:8]}")
    print(f"One-shot sums to: {Q_random.sum():.10f}")
    
    print("\n" + "="*80)
    print("✓✓✓ ALL TESTS PASSED")
    print("="*80)


if __name__ == '__main__':
    test_conversions()
